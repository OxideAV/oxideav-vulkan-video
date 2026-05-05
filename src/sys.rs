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
/// u32 (use `VK_API_VERSION_VARIANT/MAJOR/MINOR/PATCH` macros to
/// unpack — to be added in Round 2).
pub type FnVkEnumerateInstanceVersion = unsafe extern "C" fn(version: *mut u32) -> VkResult;

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
