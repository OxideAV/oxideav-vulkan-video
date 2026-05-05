//! Runtime-loaded Vulkan loader handle.
//!
//! Loaded once via `OnceLock` on first use and cached for the process
//! lifetime. If the dlopen fails the cache stores the error so
//! subsequent calls don't repeatedly hammer the dynamic linker.
//!
//! Library needed for the bridge:
//!
//! | Platform | Library                                           |
//! |----------|---------------------------------------------------|
//! | Linux    | `libvulkan.so.1`                                  |
//! | Windows  | `vulkan-1.dll` (the Khronos / LunarG loader DLL)  |
//!
//! On Windows the loader is installed by the Vulkan SDK, by GPU
//! driver packages (NVIDIA, AMD, Intel), and by Windows itself on
//! recent builds — same dlopen story as the Linux side, just a
//! different filename.
//!
//! Vulkan's bootstrap is four symbols. Every other Vulkan function
//! (including the entire `VK_KHR_video_*` extension family) is
//! reached via `vkGetInstanceProcAddr` (instance-level) or
//! `vkGetDeviceProcAddr` (device-level) after a `VkInstance` is
//! constructed. Round 1 only wires up the bootstrap; Round 2 will
//! populate the post-instance dispatch surface.

use libloading::Library;
use std::ffi::c_void;
use std::os::raw::c_char;
use std::sync::OnceLock;

// ─────────────────────────── opaque Vulkan types ─────────────────────────────

/// Vulkan instance handle. Returned by `vkCreateInstance`.
pub type VkInstance = *mut c_void;

/// Physical device handle (a GPU). Returned by
/// `vkEnumeratePhysicalDevices` (resolved post-bootstrap).
pub type VkPhysicalDevice = *mut c_void;

/// Logical device handle. Returned by `vkCreateDevice` (resolved
/// post-bootstrap).
pub type VkDevice = *mut c_void;

/// Queue handle. Returned by `vkGetDeviceQueue` (resolved
/// post-bootstrap).
pub type VkQueue = *mut c_void;

/// VkResult — return code for almost every Vulkan entry point.
pub type VkResult = i32;

/// Success status: `VK_SUCCESS == 0`.
pub const VK_SUCCESS: VkResult = 0;

/// `VkBool32` — the spec uses uint32 for booleans (1 == true, 0 ==
/// false). Used in `VkPhysicalDeviceFeatures`, `VkQueueFamilyVideo*`,
/// etc.
pub type VkBool32 = u32;

/// `VkStructureType` — the discriminant tag at the top of every
/// extensible Vulkan struct (`sType`). Drives the `pNext` chain.
pub type VkStructureType = i32;

/// `VkFlags` — generic 32-bit bitmask backing for many `*Flags`
/// typedefs in the Vulkan API.
pub type VkFlags = u32;

/// `VkInstanceCreateFlags` — reserved bitmask in `VkInstanceCreateInfo`
/// (Vulkan 1.0 has no defined bits; portability subset adds one).
pub type VkInstanceCreateFlags = VkFlags;

/// `VkQueueFlags` — bitmask of the operations supported by a queue
/// family. The bits we care about for video: `0x20` for decode and
/// `0x40` for encode.
pub type VkQueueFlags = VkFlags;

/// `VkPhysicalDeviceType` — what kind of GPU/CPU is reporting (discrete,
/// integrated, virtual, …).
pub type VkPhysicalDeviceType = i32;

// ─────────────────────────── struct-type discriminants ───────────────────────
// `sType` values for the Vulkan structs we construct here. The full
// list is in the spec / `vulkan_core.h`; we list only the ones used.

pub const VK_STRUCTURE_TYPE_APPLICATION_INFO: VkStructureType = 0;
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: VkStructureType = 1;
/// `VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2 = 1000059005`. Set on
/// the per-element `VkQueueFamilyProperties2` array passed in to
/// `vkGetPhysicalDeviceQueueFamilyProperties2` so the implementation
/// can populate the optional `pNext` chain.
pub const VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2: VkStructureType = 1000059005;
/// `VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR = 1000023012`.
/// Set on the optional `VkQueueFamilyVideoPropertiesKHR` extension
/// struct chained off `VkQueueFamilyProperties2.pNext` to retrieve the
/// `videoCodecOperations` bitmask of supported codecs per queue family.
pub const VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR: VkStructureType = 1000023012;

// ─────────────────────────── Queue flags ──────────────────────────────────────

/// `VK_QUEUE_GRAPHICS_BIT = 0x1`.
pub const VK_QUEUE_GRAPHICS_BIT: VkQueueFlags = 0x00000001;
/// `VK_QUEUE_COMPUTE_BIT = 0x2`.
pub const VK_QUEUE_COMPUTE_BIT: VkQueueFlags = 0x00000002;
/// `VK_QUEUE_TRANSFER_BIT = 0x4`.
pub const VK_QUEUE_TRANSFER_BIT: VkQueueFlags = 0x00000004;
/// `VK_QUEUE_SPARSE_BINDING_BIT = 0x8`.
pub const VK_QUEUE_SPARSE_BINDING_BIT: VkQueueFlags = 0x00000008;
/// `VK_QUEUE_PROTECTED_BIT = 0x10`.
pub const VK_QUEUE_PROTECTED_BIT: VkQueueFlags = 0x00000010;
/// `VK_QUEUE_VIDEO_DECODE_BIT_KHR = 0x20`. Indicates a queue family
/// supports `vkCmdDecodeVideoKHR`-class operations.
pub const VK_QUEUE_VIDEO_DECODE_BIT_KHR: VkQueueFlags = 0x00000020;
/// `VK_QUEUE_VIDEO_ENCODE_BIT_KHR = 0x40`. Indicates a queue family
/// supports `vkCmdEncodeVideoKHR`-class operations.
pub const VK_QUEUE_VIDEO_ENCODE_BIT_KHR: VkQueueFlags = 0x00000040;

// ─────────────────────────── Physical device type ─────────────────────────────

pub const VK_PHYSICAL_DEVICE_TYPE_OTHER: VkPhysicalDeviceType = 0;
pub const VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: VkPhysicalDeviceType = 1;
pub const VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: VkPhysicalDeviceType = 2;
pub const VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: VkPhysicalDeviceType = 3;
pub const VK_PHYSICAL_DEVICE_TYPE_CPU: VkPhysicalDeviceType = 4;

// ─────────────────────────── Vulkan version helpers ───────────────────────────

/// `VK_MAKE_API_VERSION(variant, major, minor, patch)` — pack a
/// version tuple into the 32-bit form used by `apiVersion` /
/// `applicationVersion` / `engineVersion`.
///
/// Bit layout (per the Vulkan spec): variant in bits 31..29, major in
/// 28..22, minor in 21..12, patch in 11..0.
#[inline]
pub const fn vk_make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

/// `VK_API_VERSION_1_0` packed.
pub const VK_API_VERSION_1_0: u32 = vk_make_api_version(0, 1, 0, 0);
/// `VK_API_VERSION_1_1` packed.
pub const VK_API_VERSION_1_1: u32 = vk_make_api_version(0, 1, 1, 0);
/// `VK_API_VERSION_1_2` packed.
pub const VK_API_VERSION_1_2: u32 = vk_make_api_version(0, 1, 2, 0);
/// `VK_API_VERSION_1_3` packed.
pub const VK_API_VERSION_1_3: u32 = vk_make_api_version(0, 1, 3, 0);

/// `VK_API_VERSION_VARIANT(version)` — extract the variant nibble.
#[inline]
pub const fn vk_api_version_variant(v: u32) -> u32 {
    v >> 29
}
/// `VK_API_VERSION_MAJOR(version)` — extract the major component.
#[inline]
pub const fn vk_api_version_major(v: u32) -> u32 {
    (v >> 22) & 0x7F
}
/// `VK_API_VERSION_MINOR(version)` — extract the minor component.
#[inline]
pub const fn vk_api_version_minor(v: u32) -> u32 {
    (v >> 12) & 0x3FF
}
/// `VK_API_VERSION_PATCH(version)` — extract the patch component.
#[inline]
pub const fn vk_api_version_patch(v: u32) -> u32 {
    v & 0xFFF
}

// ─────────────────────────── Spec sizes ───────────────────────────────────────

/// `VK_MAX_PHYSICAL_DEVICE_NAME_SIZE = 256`.
pub const VK_MAX_PHYSICAL_DEVICE_NAME_SIZE: usize = 256;
/// `VK_UUID_SIZE = 16`.
pub const VK_UUID_SIZE: usize = 16;
/// `VK_MAX_EXTENSION_NAME_SIZE = 256`.
pub const VK_MAX_EXTENSION_NAME_SIZE: usize = 256;
/// `VK_MAX_MEMORY_TYPES = 32`.
pub const VK_MAX_MEMORY_TYPES: usize = 32;
/// `VK_MAX_MEMORY_HEAPS = 16`.
pub const VK_MAX_MEMORY_HEAPS: usize = 16;

/// Generic Vulkan function pointer returned by
/// `vkGetInstanceProcAddr` / `vkGetDeviceProcAddr`. Caller transmutes
/// to the specific signature for the function being resolved.
///
/// The `PFN_` prefix and the lower-case `vk` are the canonical Vulkan
/// spelling — we keep them rather than the Rust `UpperCamelCase`
/// rename, so a header-style search across the spec docs and our
/// bridge produces a hit.
#[allow(non_camel_case_types)]
pub type PFN_vkVoidFunction = Option<unsafe extern "C" fn()>;

// ─────────────────────────── function pointer types ──────────────────────────

/// `vkGetInstanceProcAddr(instance, name)` — the universal Vulkan
/// dispatch entry. Pass a null `instance` for the platform-level
/// entries (`vkCreateInstance`, `vkEnumerateInstance*`) and a real
/// `VkInstance` for instance-level entries.
pub type FnVkGetInstanceProcAddr =
    unsafe extern "C" fn(instance: VkInstance, name: *const c_char) -> PFN_vkVoidFunction;

/// `vkCreateInstance(create_info, allocator, instance_out)` — needed
/// to construct the `VkInstance` that subsequent
/// `vkGetInstanceProcAddr` calls operate against. The `create_info`
/// struct is large and not modelled in Round 1.
pub type FnVkCreateInstance = unsafe extern "C" fn(
    create_info: *const c_void,
    allocator: *const c_void,
    instance: *mut VkInstance,
) -> VkResult;

/// `vkEnumerateInstanceExtensionProperties(layer_name, count,
/// properties)` — used to verify the loader exposes
/// `VK_KHR_video_queue` and friends. The `properties` struct is
/// modelled in Round 2.
pub type FnVkEnumerateInstanceExtensionProperties = unsafe extern "C" fn(
    layer_name: *const c_char,
    property_count: *mut u32,
    properties: *mut c_void,
) -> VkResult;

/// `vkEnumerateInstanceVersion(version)` — Vulkan 1.1+ runtime sanity
/// check. Returns the loader's reported `apiVersion` packed into a
/// u32 (use `vk_api_version_*` accessor functions to unpack).
pub type FnVkEnumerateInstanceVersion = unsafe extern "C" fn(version: *mut u32) -> VkResult;

// ─────────────────────────── Vulkan core structs ──────────────────────────────
//
// Layouts mirror the Khronos `vulkan_core.h` definitions. The Vulkan
// spec is the canonical source for these — the C header is just a
// vendor-supplied translation. We keep the field names in
// snake_case (Rust idiom) but the byte layout is identical.

/// `VkApplicationInfo` — describes the application + requested API
/// version, passed to `vkCreateInstance` via
/// `VkInstanceCreateInfo.pApplicationInfo`.
#[repr(C)]
#[derive(Debug)]
pub struct VkApplicationInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub p_application_name: *const c_char,
    pub application_version: u32,
    pub p_engine_name: *const c_char,
    pub engine_version: u32,
    pub api_version: u32,
}

/// `VkInstanceCreateInfo` — argument bundle for `vkCreateInstance`.
#[repr(C)]
#[derive(Debug)]
pub struct VkInstanceCreateInfo {
    pub s_type: VkStructureType,
    pub p_next: *const c_void,
    pub flags: VkInstanceCreateFlags,
    pub p_application_info: *const VkApplicationInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
}

/// `VkExtensionProperties` — fixed-size record returned by
/// `vkEnumerateInstanceExtensionProperties` and
/// `vkEnumerateDeviceExtensionProperties`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkExtensionProperties {
    pub extension_name: [c_char; VK_MAX_EXTENSION_NAME_SIZE],
    pub spec_version: u32,
}

/// `VkPhysicalDeviceSparseProperties` — substruct nested inside
/// `VkPhysicalDeviceProperties`. We don't read any of these fields,
/// but the layout has to match for the parent struct's offsets to
/// land where the driver writes them.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VkPhysicalDeviceSparseProperties {
    pub residency_standard_2d_block_shape: VkBool32,
    pub residency_standard_2d_multisample_block_shape: VkBool32,
    pub residency_standard_3d_block_shape: VkBool32,
    pub residency_aligned_mip_size: VkBool32,
    pub residency_non_resident_strict: VkBool32,
}

/// `VkPhysicalDeviceLimits` — large substruct nested inside
/// `VkPhysicalDeviceProperties`. We don't expose any limit fields in
/// Round 2 but the byte layout has to match the spec so the trailing
/// `sparseProperties` field of the parent lands at the right offset.
///
/// Field types and ordering mirror the Vulkan 1.0 spec exactly; the
/// struct is forward-compatible (newer Vulkan versions add
/// `VkPhysicalDeviceProperties2` with a `pNext` chain rather than
/// extending this struct in-place).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkPhysicalDeviceLimits {
    pub max_image_dimension_1d: u32,
    pub max_image_dimension_2d: u32,
    pub max_image_dimension_3d: u32,
    pub max_image_dimension_cube: u32,
    pub max_image_array_layers: u32,
    pub max_texel_buffer_elements: u32,
    pub max_uniform_buffer_range: u32,
    pub max_storage_buffer_range: u32,
    pub max_push_constants_size: u32,
    pub max_memory_allocation_count: u32,
    pub max_sampler_allocation_count: u32,
    pub buffer_image_granularity: u64,
    pub sparse_address_space_size: u64,
    pub max_bound_descriptor_sets: u32,
    pub max_per_stage_descriptor_samplers: u32,
    pub max_per_stage_descriptor_uniform_buffers: u32,
    pub max_per_stage_descriptor_storage_buffers: u32,
    pub max_per_stage_descriptor_sampled_images: u32,
    pub max_per_stage_descriptor_storage_images: u32,
    pub max_per_stage_descriptor_input_attachments: u32,
    pub max_per_stage_resources: u32,
    pub max_descriptor_set_samplers: u32,
    pub max_descriptor_set_uniform_buffers: u32,
    pub max_descriptor_set_uniform_buffers_dynamic: u32,
    pub max_descriptor_set_storage_buffers: u32,
    pub max_descriptor_set_storage_buffers_dynamic: u32,
    pub max_descriptor_set_sampled_images: u32,
    pub max_descriptor_set_storage_images: u32,
    pub max_descriptor_set_input_attachments: u32,
    pub max_vertex_input_attributes: u32,
    pub max_vertex_input_bindings: u32,
    pub max_vertex_input_attribute_offset: u32,
    pub max_vertex_input_binding_stride: u32,
    pub max_vertex_output_components: u32,
    pub max_tessellation_generation_level: u32,
    pub max_tessellation_patch_size: u32,
    pub max_tessellation_control_per_vertex_input_components: u32,
    pub max_tessellation_control_per_vertex_output_components: u32,
    pub max_tessellation_control_per_patch_output_components: u32,
    pub max_tessellation_control_total_output_components: u32,
    pub max_tessellation_evaluation_input_components: u32,
    pub max_tessellation_evaluation_output_components: u32,
    pub max_geometry_shader_invocations: u32,
    pub max_geometry_input_components: u32,
    pub max_geometry_output_components: u32,
    pub max_geometry_output_vertices: u32,
    pub max_geometry_total_output_components: u32,
    pub max_fragment_input_components: u32,
    pub max_fragment_output_attachments: u32,
    pub max_fragment_dual_src_attachments: u32,
    pub max_fragment_combined_output_resources: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_work_group_size: [u32; 3],
    pub sub_pixel_precision_bits: u32,
    pub sub_texel_precision_bits: u32,
    pub mipmap_precision_bits: u32,
    pub max_draw_indexed_index_value: u32,
    pub max_draw_indirect_count: u32,
    pub max_sampler_lod_bias: f32,
    pub max_sampler_anisotropy: f32,
    pub max_viewports: u32,
    pub max_viewport_dimensions: [u32; 2],
    pub viewport_bounds_range: [f32; 2],
    pub viewport_sub_pixel_bits: u32,
    pub min_memory_map_alignment: usize,
    pub min_texel_buffer_offset_alignment: u64,
    pub min_uniform_buffer_offset_alignment: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub min_texel_offset: i32,
    pub max_texel_offset: u32,
    pub min_texel_gather_offset: i32,
    pub max_texel_gather_offset: u32,
    pub min_interpolation_offset: f32,
    pub max_interpolation_offset: f32,
    pub sub_pixel_interpolation_offset_bits: u32,
    pub max_framebuffer_width: u32,
    pub max_framebuffer_height: u32,
    pub max_framebuffer_layers: u32,
    pub framebuffer_color_sample_counts: VkFlags,
    pub framebuffer_depth_sample_counts: VkFlags,
    pub framebuffer_stencil_sample_counts: VkFlags,
    pub framebuffer_no_attachments_sample_counts: VkFlags,
    pub max_color_attachments: u32,
    pub sampled_image_color_sample_counts: VkFlags,
    pub sampled_image_integer_sample_counts: VkFlags,
    pub sampled_image_depth_sample_counts: VkFlags,
    pub sampled_image_stencil_sample_counts: VkFlags,
    pub storage_image_sample_counts: VkFlags,
    pub max_sample_mask_words: u32,
    pub timestamp_compute_and_graphics: VkBool32,
    pub timestamp_period: f32,
    pub max_clip_distances: u32,
    pub max_cull_distances: u32,
    pub max_combined_clip_and_cull_distances: u32,
    pub discrete_queue_priorities: u32,
    pub point_size_range: [f32; 2],
    pub line_width_range: [f32; 2],
    pub point_size_granularity: f32,
    pub line_width_granularity: f32,
    pub strict_lines: VkBool32,
    pub standard_sample_locations: VkBool32,
    pub optimal_buffer_copy_offset_alignment: u64,
    pub optimal_buffer_copy_row_pitch_alignment: u64,
    pub non_coherent_atom_size: u64,
}

/// `VkPhysicalDeviceProperties` — populated by
/// `vkGetPhysicalDeviceProperties`. The Vulkan 1.0 layout is forward
/// compatible: newer Vulkan versions surface additional information
/// via `VkPhysicalDeviceProperties2`'s `pNext` chain rather than
/// modifying this struct.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkPhysicalDeviceProperties {
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: VkPhysicalDeviceType,
    pub device_name: [c_char; VK_MAX_PHYSICAL_DEVICE_NAME_SIZE],
    pub pipeline_cache_uuid: [u8; VK_UUID_SIZE],
    pub limits: VkPhysicalDeviceLimits,
    pub sparse_properties: VkPhysicalDeviceSparseProperties,
}

/// `VkExtent3D` — used in `VkQueueFamilyProperties`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VkExtent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

/// `VkQueueFamilyProperties` — substruct of `VkQueueFamilyProperties2`.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VkQueueFamilyProperties {
    pub queue_flags: VkQueueFlags,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: VkExtent3D,
}

/// `VkQueueFamilyProperties2` — populated by
/// `vkGetPhysicalDeviceQueueFamilyProperties2`. The `p_next` chain may
/// carry a `VkQueueFamilyVideoPropertiesKHR` to surface per-queue
/// `videoCodecOperations` when `VK_KHR_video_queue` is present.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VkQueueFamilyProperties2 {
    pub s_type: VkStructureType,
    pub p_next: *mut c_void,
    pub queue_family_properties: VkQueueFamilyProperties,
}

/// `VkQueueFamilyVideoPropertiesKHR` — extension struct chained off
/// `VkQueueFamilyProperties2.pNext` to expose the codec operation
/// bitmask the queue family supports (decode H.264, decode HEVC,
/// encode AV1, …).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct VkQueueFamilyVideoPropertiesKHR {
    pub s_type: VkStructureType,
    pub p_next: *mut c_void,
    pub video_codec_operations: VkFlags,
}

// ─────────────────────────── Post-bootstrap function pointer types ────────────

/// `vkDestroyInstance(instance, allocator)` — called from
/// `Drop for Instance` to release the instance handle.
pub type FnVkDestroyInstance =
    unsafe extern "C" fn(instance: VkInstance, allocator: *const c_void);

/// `vkEnumeratePhysicalDevices(instance, count, devices)` — populates
/// the array of `VkPhysicalDevice` handles. Two-call pattern: pass
/// null `devices` to query the count, then again with a sized buffer.
pub type FnVkEnumeratePhysicalDevices = unsafe extern "C" fn(
    instance: VkInstance,
    physical_device_count: *mut u32,
    physical_devices: *mut VkPhysicalDevice,
) -> VkResult;

/// `vkGetPhysicalDeviceProperties(physical_device, properties_out)` —
/// fills the (large) `VkPhysicalDeviceProperties` record.
pub type FnVkGetPhysicalDeviceProperties = unsafe extern "C" fn(
    physical_device: VkPhysicalDevice,
    properties: *mut VkPhysicalDeviceProperties,
);

/// `vkEnumerateDeviceExtensionProperties(physical_device, layer_name,
/// count, properties)` — two-call pattern returning the extensions a
/// physical device supports (e.g. `VK_KHR_video_decode_h264`).
pub type FnVkEnumerateDeviceExtensionProperties = unsafe extern "C" fn(
    physical_device: VkPhysicalDevice,
    layer_name: *const c_char,
    property_count: *mut u32,
    properties: *mut VkExtensionProperties,
) -> VkResult;

/// `vkGetPhysicalDeviceQueueFamilyProperties2(physical_device, count,
/// properties)` — Vulkan 1.1 / `VK_KHR_get_physical_device_properties2`
/// form. The `_2` form gives us a `pNext` chain so we can request a
/// `VkQueueFamilyVideoPropertiesKHR` extension struct in a single
/// driver round-trip.
pub type FnVkGetPhysicalDeviceQueueFamilyProperties2 = unsafe extern "C" fn(
    physical_device: VkPhysicalDevice,
    queue_family_property_count: *mut u32,
    queue_family_properties: *mut VkQueueFamilyProperties2,
);

// ─────────────────────────── Vtable ───────────────────────────────────────────

/// Resolved function pointers for the bootstrap Vulkan symbol set.
///
/// All fields are `unsafe extern "C" fn(...)` pointer types — callers
/// are responsible for the FFI invariants (correct argument types,
/// instance lifetime, `VkResult` checking).
pub struct Vtable {
    pub vk_get_instance_proc_addr: FnVkGetInstanceProcAddr,
    pub vk_create_instance: FnVkCreateInstance,
    pub vk_enumerate_instance_extension_properties: FnVkEnumerateInstanceExtensionProperties,
    pub vk_enumerate_instance_version: FnVkEnumerateInstanceVersion,
    // Keep library alive
    _libvulkan: Library,
}

/// Smoke-test wrapper used by tests + by the pre-flight load check
/// in `register()`. Holds the raw `Library` handle so callers can
/// assert that dlopen succeeded without paying the full dlsym tour.
pub struct FrameworkSmoke {
    pub libvulkan: Library,
}

// ─────────────────────────── Caches ───────────────────────────────────────────

static VTABLE: OnceLock<Result<Vtable, String>> = OnceLock::new();
static FRAMEWORK: OnceLock<Result<FrameworkSmoke, String>> = OnceLock::new();

/// Get (or load) the fully-resolved vtable. Returns the cached `Err`
/// if a previous load attempt failed.
pub fn vtable() -> Result<&'static Vtable, &'static str> {
    VTABLE
        .get_or_init(load_vtable)
        .as_ref()
        .map_err(|s| s.as_str())
}

/// Cheap framework-load check used by `register()`. Resolves the
/// loader but does no dlsym work.
pub fn framework() -> Result<&'static FrameworkSmoke, &'static str> {
    FRAMEWORK
        .get_or_init(load_smoke)
        .as_ref()
        .map_err(|s| s.as_str())
}

/// Per-platform soname / dll filename for the Vulkan loader.
///
/// Linux uses `libvulkan.so.1` (the SONAME shipped by the Khronos
/// loader and by every distro package). Windows uses `vulkan-1.dll`
/// (the standard filename installed by the Vulkan SDK, by GPU
/// drivers, and by Windows itself on recent builds).
#[cfg(target_os = "linux")]
const VULKAN_LIBRARY: &str = "libvulkan.so.1";
#[cfg(target_os = "windows")]
const VULKAN_LIBRARY: &str = "vulkan-1.dll";

fn load_smoke() -> Result<FrameworkSmoke, String> {
    Ok(FrameworkSmoke {
        libvulkan: open(VULKAN_LIBRARY)?,
    })
}

fn load_vtable() -> Result<Vtable, String> {
    let libvulkan = open(VULKAN_LIBRARY)?;

    macro_rules! sym {
        ($lib:expr, $name:expr, $ty:ty) => {{
            let s: libloading::Symbol<$ty> = unsafe {
                $lib.get(concat!($name, "\0").as_bytes())
                    .map_err(|e| format!("dlsym {}: {}", $name, e))?
            };
            *s
        }};
    }

    Ok(Vtable {
        vk_get_instance_proc_addr: sym!(
            libvulkan,
            "vkGetInstanceProcAddr",
            FnVkGetInstanceProcAddr
        ),
        vk_create_instance: sym!(libvulkan, "vkCreateInstance", FnVkCreateInstance),
        vk_enumerate_instance_extension_properties: sym!(
            libvulkan,
            "vkEnumerateInstanceExtensionProperties",
            FnVkEnumerateInstanceExtensionProperties
        ),
        vk_enumerate_instance_version: sym!(
            libvulkan,
            "vkEnumerateInstanceVersion",
            FnVkEnumerateInstanceVersion
        ),
        _libvulkan: libvulkan,
    })
}

fn open(path: &str) -> Result<Library, String> {
    // SAFETY: dlopen on a soname with no init callbacks; equivalent to
    // a normal program startup load.
    unsafe { Library::new(path) }.map_err(|e| format!("dlopen {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: libvulkan.so.1 on this machine loads cleanly.
    #[test]
    fn frameworks_load() {
        let fw = framework().expect("framework load");
        // Confirm the bootstrap entry is present.
        let _: libloading::Symbol<unsafe extern "C" fn()> = unsafe {
            fw.libvulkan
                .get(b"vkGetInstanceProcAddr\0")
                .expect("vkGetInstanceProcAddr symbol")
        };
    }

    /// Verify the vtable resolves all symbols.
    #[test]
    fn vtable_resolves() {
        vtable().expect("vtable load");
    }
}
