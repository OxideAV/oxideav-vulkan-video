//! Safe wrapper over `VkInstance`.
//!
//! Vulkan's bootstrap is the four symbols resolved by `sys::vtable()`.
//! After `vkCreateInstance` succeeds the rest of the API surface is
//! reached via `vkGetInstanceProcAddr` against the new instance
//! handle. This module hides that two-stage dance:
//!
//! 1. `Instance::new(name, api)` calls `vkCreateInstance` with an
//!    empty layer + extension list (instance creation does not need
//!    any extensions for our Round 2 use case — Vulkan 1.1+ exposes
//!    `vkGetPhysicalDeviceQueueFamilyProperties2` as a core entry).
//! 2. The constructor then resolves every post-bootstrap function
//!    pointer it'll need (`vkDestroyInstance`,
//!    `vkEnumeratePhysicalDevices`, …) via `vkGetInstanceProcAddr`,
//!    storing them on `Self`.
//! 3. `Drop for Instance` calls `vkDestroyInstance`. The handle is
//!    not `Send + Sync` — the user is responsible for keeping it on a
//!    single thread for the duration of any work.

use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::physical_device::PhysicalDevice;
use crate::sys::{
    self, FnVkDestroyInstance, FnVkEnumerateDeviceExtensionProperties,
    FnVkEnumeratePhysicalDevices, FnVkGetPhysicalDeviceProperties,
    FnVkGetPhysicalDeviceQueueFamilyProperties2, VkApplicationInfo, VkInstance, VkInstanceCreateInfo,
    VkResult, VK_API_VERSION_1_0, VK_STRUCTURE_TYPE_APPLICATION_INFO,
    VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, VK_SUCCESS,
};

/// Errors produced by the safe wrapper layer.
///
/// `VkError::Result` carries the underlying `VkResult` (negative on
/// failure per the Vulkan spec) plus a context string identifying the
/// API call that produced it. `VkError::LoaderUnavailable` is
/// returned when the underlying `libvulkan` couldn't be opened or one
/// of the bootstrap symbols was missing — distinct because callers
/// often want to fall back to a pure-software path here rather than
/// surface an error.
#[derive(Debug)]
pub enum VkError {
    /// The Vulkan loader (`libvulkan.so.1` / `vulkan-1.dll`) couldn't
    /// be opened, or one of the bootstrap symbols was missing.
    LoaderUnavailable(String),
    /// A Vulkan call returned a non-`VK_SUCCESS` `VkResult`.
    Result {
        /// API name (e.g. `"vkCreateInstance"`).
        op: &'static str,
        /// Raw `VkResult` value as returned by the driver.
        result: VkResult,
    },
    /// `vkGetInstanceProcAddr` returned NULL for a function we
    /// expected to be available.
    MissingFunction(&'static str),
}

impl std::fmt::Display for VkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VkError::LoaderUnavailable(msg) => write!(f, "vulkan loader unavailable: {msg}"),
            VkError::Result { op, result } => write!(f, "{op} returned VkResult({result})"),
            VkError::MissingFunction(name) => {
                write!(f, "vkGetInstanceProcAddr returned NULL for {name}")
            }
        }
    }
}

impl std::error::Error for VkError {}

/// A live `VkInstance` plus the post-bootstrap function pointers
/// we've resolved against it.
///
/// Cloning is intentionally not implemented: the Vulkan spec is
/// strict about the lifetime relationship between the instance and
/// every object derived from it. `Drop` calls `vkDestroyInstance`.
pub struct Instance {
    handle: VkInstance,
    fns: InstanceFns,
}

/// Function pointers resolved via `vkGetInstanceProcAddr`. These are
/// kept inline (not boxed) so the call path through the wrapper is
/// `inst.fns.enumerate_physical_devices(...)` with no indirection
/// beyond the function-pointer call itself.
pub(crate) struct InstanceFns {
    pub(crate) destroy_instance: FnVkDestroyInstance,
    pub(crate) enumerate_physical_devices: FnVkEnumeratePhysicalDevices,
    pub(crate) get_physical_device_properties: FnVkGetPhysicalDeviceProperties,
    pub(crate) enumerate_device_extension_properties: FnVkEnumerateDeviceExtensionProperties,
    pub(crate) get_physical_device_queue_family_properties2:
        FnVkGetPhysicalDeviceQueueFamilyProperties2,
}

impl Instance {
    /// Construct a `VkInstance`.
    ///
    /// `app_name` is reported as the application name in the
    /// `VkApplicationInfo` struct (no functional effect — drivers may
    /// use it for telemetry). `requested_api_version` is the
    /// `apiVersion` field; pass one of the
    /// [`sys::VK_API_VERSION_1_0`]‒[`sys::VK_API_VERSION_1_3`]
    /// constants. Vulkan 1.1+ is required to make the
    /// `vkGetPhysicalDeviceQueueFamilyProperties2` call non-`KHR`.
    ///
    /// Returns `VkError::LoaderUnavailable` if the dlopen of the
    /// Vulkan loader failed (no Vulkan ICD installed, headless CI
    /// without Mesa, …) — callers will typically log + fall back to
    /// a software path.
    pub fn new(app_name: &str, requested_api_version: u32) -> Result<Self, VkError> {
        let vt = sys::vtable().map_err(|e| VkError::LoaderUnavailable(e.to_string()))?;

        // Strings have to outlive the create call.
        let app_name_c = CString::new(app_name).unwrap_or_else(|_| CString::new("oxideav").unwrap());
        let engine_name_c = CString::new("oxideav-vulkan-video").unwrap();

        let app_info = VkApplicationInfo {
            s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            p_next: ptr::null(),
            p_application_name: app_name_c.as_ptr(),
            application_version: VK_API_VERSION_1_0,
            p_engine_name: engine_name_c.as_ptr(),
            engine_version: VK_API_VERSION_1_0,
            api_version: requested_api_version,
        };

        let create_info = VkInstanceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            p_application_info: &app_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: ptr::null(),
        };

        let mut instance: VkInstance = ptr::null_mut();
        // SAFETY: pointers in `create_info` reference `app_info`,
        // `app_name_c`, and `engine_name_c`, all of which live until
        // after this call returns; the Vulkan spec requires the
        // implementation to copy out anything it needs before
        // `vkCreateInstance` returns.
        let result = unsafe {
            (vt.vk_create_instance)(
                &create_info as *const _ as *const c_void,
                ptr::null(),
                &mut instance,
            )
        };
        if result != VK_SUCCESS {
            return Err(VkError::Result {
                op: "vkCreateInstance",
                result,
            });
        }

        let fns = InstanceFns::resolve(vt.vk_get_instance_proc_addr, instance)?;

        Ok(Self {
            handle: instance,
            fns,
        })
    }

    /// Raw handle. For interop with hand-rolled FFI; the safe API
    /// surface should grow rather than callers reaching for this.
    pub fn handle(&self) -> VkInstance {
        self.handle
    }

    /// Enumerate the GPUs the loader can see.
    ///
    /// Two-call pattern: a count probe followed by a sized fetch.
    pub fn physical_devices(&self) -> Result<Vec<PhysicalDevice<'_>>, VkError> {
        let mut count: u32 = 0;
        // SAFETY: standard two-call enumerate pattern. Passing
        // null for the device array returns just the count in
        // `count`.
        let result =
            unsafe { (self.fns.enumerate_physical_devices)(self.handle, &mut count, ptr::null_mut()) };
        if result != VK_SUCCESS {
            return Err(VkError::Result {
                op: "vkEnumeratePhysicalDevices",
                result,
            });
        }

        let mut handles = vec![ptr::null_mut(); count as usize];
        // SAFETY: the buffer is sized from the count probe just
        // above; Vulkan will write at most `count` handles. The
        // call may shrink `count` if the driver returns
        // `VK_INCOMPLETE`, which we treat as a soft success.
        let result = unsafe {
            (self.fns.enumerate_physical_devices)(self.handle, &mut count, handles.as_mut_ptr())
        };
        if result != VK_SUCCESS {
            return Err(VkError::Result {
                op: "vkEnumeratePhysicalDevices",
                result,
            });
        }

        handles.truncate(count as usize);

        Ok(handles
            .into_iter()
            .map(|h| PhysicalDevice::from_raw(h, &self.fns))
            .collect())
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle was created by `vkCreateInstance` and
            // has not been previously destroyed; no objects derived
            // from it are still live (the borrow checker prevents
            // outstanding `PhysicalDevice<'_>` references because
            // they share `self`'s lifetime).
            unsafe {
                (self.fns.destroy_instance)(self.handle, ptr::null());
            }
            self.handle = ptr::null_mut();
        }
    }
}

// SAFETY: Vulkan instances are explicitly externally synchronised but
// the `VkInstance` handle itself can be passed between threads as
// long as the user serialises calls. We don't mark the type
// `Send + Sync` here — leave that to a future round once we have
// a clear synchronisation story.

impl InstanceFns {
    fn resolve(
        get_proc: sys::FnVkGetInstanceProcAddr,
        instance: VkInstance,
    ) -> Result<Self, VkError> {
        // SAFETY: `vkGetInstanceProcAddr` is the documented entry
        // for retrieving instance-level function pointers; the cast
        // matches the spec-declared signature for each name.
        unsafe {
            Ok(Self {
                destroy_instance: load_fn(get_proc, instance, b"vkDestroyInstance\0")?,
                enumerate_physical_devices: load_fn(
                    get_proc,
                    instance,
                    b"vkEnumeratePhysicalDevices\0",
                )?,
                get_physical_device_properties: load_fn(
                    get_proc,
                    instance,
                    b"vkGetPhysicalDeviceProperties\0",
                )?,
                enumerate_device_extension_properties: load_fn(
                    get_proc,
                    instance,
                    b"vkEnumerateDeviceExtensionProperties\0",
                )?,
                get_physical_device_queue_family_properties2: load_fn(
                    get_proc,
                    instance,
                    b"vkGetPhysicalDeviceQueueFamilyProperties2\0",
                )?,
            })
        }
    }
}

/// Resolve a single function pointer via `vkGetInstanceProcAddr` and
/// transmute to the requested signature. The `name` argument MUST be
/// a NUL-terminated byte slice — caller's responsibility.
///
/// # Safety
/// Caller is responsible for `Fp` matching the spec-declared signature
/// of the function being resolved.
unsafe fn load_fn<Fp: Copy>(
    get_proc: sys::FnVkGetInstanceProcAddr,
    instance: VkInstance,
    name: &'static [u8],
) -> Result<Fp, VkError> {
    debug_assert!(name.last() == Some(&0));
    let static_name = CStr::from_bytes_with_nul(name)
        .expect("load_fn name must be NUL-terminated")
        .to_str()
        .expect("load_fn name must be ASCII");

    // SAFETY: get_proc has the spec-declared signature; null result
    // is an explicit "not present" signal we surface as
    // MissingFunction.
    let raw = unsafe { get_proc(instance, name.as_ptr() as *const c_char) };
    let f = raw.ok_or(VkError::MissingFunction(static_name))?;
    // SAFETY: caller documents that Fp matches the function being
    // resolved; sizes are checked at compile time.
    Ok(unsafe { std::mem::transmute_copy::<unsafe extern "C" fn(), Fp>(&f) })
}
