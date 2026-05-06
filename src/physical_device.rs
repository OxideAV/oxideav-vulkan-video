//! Safe wrapper over `VkPhysicalDevice`.
//!
//! A `PhysicalDevice` borrows from the parent `Instance` (via
//! `&InstanceFns`), so the borrow checker enforces the spec rule that
//! a physical-device handle must not outlive the instance it was
//! enumerated from.
//!
//! Round 2 surface: `properties()`, `extension_names()`, and the
//! convenience `supports_video_extensions()` summary built on top of
//! the latter. `video_queue_family_indices()` exercises the
//! `vkGetPhysicalDeviceQueueFamilyProperties2` chain to identify
//! which queue families on the device can run video decode/encode
//! commands.

use std::ffi::c_void;
use std::ptr;

use crate::instance::{InstanceFns, VkError};
use crate::sys::{
    VkPhysicalDevice, VkPhysicalDeviceProperties, VkPhysicalDeviceType, VkQueueFamilyProperties2,
    VkQueueFamilyVideoPropertiesKHR, VK_PHYSICAL_DEVICE_TYPE_CPU,
    VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
    VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU, VK_QUEUE_VIDEO_DECODE_BIT_KHR,
    VK_QUEUE_VIDEO_ENCODE_BIT_KHR, VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
    VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR, VK_SUCCESS,
};

// ─────────────────────────── extension name constants ────────────────────────

/// `VK_KHR_video_queue` — the umbrella extension required by every
/// video decode/encode codec extension below it.
pub const VK_KHR_VIDEO_QUEUE_NAME: &str = "VK_KHR_video_queue";
pub const VK_KHR_VIDEO_DECODE_QUEUE_NAME: &str = "VK_KHR_video_decode_queue";
pub const VK_KHR_VIDEO_DECODE_H264_NAME: &str = "VK_KHR_video_decode_h264";
pub const VK_KHR_VIDEO_DECODE_H265_NAME: &str = "VK_KHR_video_decode_h265";
pub const VK_KHR_VIDEO_DECODE_AV1_NAME: &str = "VK_KHR_video_decode_av1";
pub const VK_KHR_VIDEO_ENCODE_QUEUE_NAME: &str = "VK_KHR_video_encode_queue";
pub const VK_KHR_VIDEO_ENCODE_H264_NAME: &str = "VK_KHR_video_encode_h264";
pub const VK_KHR_VIDEO_ENCODE_H265_NAME: &str = "VK_KHR_video_encode_h265";

// ─────────────────────────── PhysicalDeviceProperties (subset) ───────────────

/// Subset of `VkPhysicalDeviceProperties` returned to safe Rust.
#[derive(Debug, Clone)]
pub struct PhysicalDeviceProperties {
    /// `deviceName` decoded from the C buffer (lossy UTF-8).
    pub name: String,
    /// `vendorID` (PCI vendor; 0x10DE = NVIDIA, 0x1002 = AMD,
    /// 0x8086 = Intel, …).
    pub vendor_id: u32,
    /// `deviceID` (PCI device id).
    pub device_id: u32,
    /// `deviceType` decoded into a Rust enum.
    pub device_type: PhysicalDeviceType,
    /// `apiVersion` packed (use `vk_api_version_*` accessors).
    pub api_version: u32,
    /// `driverVersion` — encoding is vendor-specific.
    pub driver_version: u32,
}

/// Decoded form of `VkPhysicalDeviceType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalDeviceType {
    Other,
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
}

impl From<VkPhysicalDeviceType> for PhysicalDeviceType {
    fn from(v: VkPhysicalDeviceType) -> Self {
        match v {
            VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => PhysicalDeviceType::IntegratedGpu,
            VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => PhysicalDeviceType::DiscreteGpu,
            VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => PhysicalDeviceType::VirtualGpu,
            VK_PHYSICAL_DEVICE_TYPE_CPU => PhysicalDeviceType::Cpu,
            _ => PhysicalDeviceType::Other,
        }
    }
}

// ─────────────────────────── VideoExtensionSupport summary ───────────────────

/// Per-codec support summary derived from the device's extension
/// list.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VideoExtensionSupport {
    /// `VK_KHR_video_queue`.
    pub queue_khr: bool,
    /// `VK_KHR_video_decode_h264`.
    pub decode_h264: bool,
    /// `VK_KHR_video_decode_h265`.
    pub decode_h265: bool,
    /// `VK_KHR_video_decode_av1`.
    pub decode_av1: bool,
    /// `VK_KHR_video_encode_h264`.
    pub encode_h264: bool,
    /// `VK_KHR_video_encode_h265`.
    pub encode_h265: bool,
}

// ─────────────────────────── PhysicalDevice ──────────────────────────────────

/// A `VkPhysicalDevice` bound to its parent `Instance` lifetime.
///
/// Constructed via [`crate::Instance::physical_devices`]. Cheap to
/// hold (a raw pointer + a `&InstanceFns` reference); the underlying
/// handle is owned by the implementation, not freed by `Drop` here.
pub struct PhysicalDevice<'instance> {
    handle: VkPhysicalDevice,
    fns: &'instance InstanceFns,
}

impl<'instance> PhysicalDevice<'instance> {
    pub(crate) fn from_raw(handle: VkPhysicalDevice, fns: &'instance InstanceFns) -> Self {
        Self { handle, fns }
    }

    /// Raw handle. Mostly for interop / tests.
    pub fn handle(&self) -> VkPhysicalDevice {
        self.handle
    }

    /// Crate-internal: borrow the parent instance's dispatch table.
    /// Used by [`crate::Device::new`] (which needs `vkCreateDevice` +
    /// `vkGetInstanceProcAddr`) and [`crate::video`] (memory-type
    /// lookup, video-capability query).
    pub(crate) fn instance_fns(&self) -> &'instance InstanceFns {
        self.fns
    }

    /// Pull the (Vulkan 1.0) properties record. The driver fills the
    /// large `VkPhysicalDeviceProperties` struct on the stack here;
    /// we copy out only the fields oxideav cares about and drop the
    /// rest.
    pub fn properties(&self) -> PhysicalDeviceProperties {
        // SAFETY: zero-initialised storage of the right size+layout
        // for `VkPhysicalDeviceProperties`. Vulkan writes every
        // field; nothing reads the buffer before the call.
        let mut props: VkPhysicalDeviceProperties = unsafe { std::mem::zeroed() };
        unsafe {
            (self.fns.get_physical_device_properties)(self.handle, &mut props);
        }
        PhysicalDeviceProperties {
            name: c_chars_to_string(&props.device_name),
            vendor_id: props.vendor_id,
            device_id: props.device_id,
            device_type: props.device_type.into(),
            api_version: props.api_version,
            driver_version: props.driver_version,
        }
    }

    /// Enumerate the device extensions advertised by this physical
    /// device.
    ///
    /// Two-call pattern: count probe, then sized fetch. Each entry
    /// is decoded from the fixed 256-byte `extensionName` buffer.
    pub fn extension_names(&self) -> Result<Vec<String>, VkError> {
        let mut count: u32 = 0;
        // SAFETY: passing null for `properties` makes Vulkan write
        // only the count.
        let result = unsafe {
            (self.fns.enumerate_device_extension_properties)(
                self.handle,
                ptr::null(),
                &mut count,
                ptr::null_mut(),
            )
        };
        if result != VK_SUCCESS {
            return Err(VkError::Result {
                op: "vkEnumerateDeviceExtensionProperties",
                result,
            });
        }

        let mut buf = vec![
            crate::sys::VkExtensionProperties {
                extension_name: [0; crate::sys::VK_MAX_EXTENSION_NAME_SIZE],
                spec_version: 0,
            };
            count as usize
        ];
        // SAFETY: buf is sized from the probe; Vulkan writes at most
        // `count` entries. The two-call pattern is the standard
        // Vulkan idiom.
        let result = unsafe {
            (self.fns.enumerate_device_extension_properties)(
                self.handle,
                ptr::null(),
                &mut count,
                buf.as_mut_ptr(),
            )
        };
        if result != VK_SUCCESS {
            return Err(VkError::Result {
                op: "vkEnumerateDeviceExtensionProperties",
                result,
            });
        }
        buf.truncate(count as usize);

        Ok(buf
            .iter()
            .map(|p| c_chars_to_string(&p.extension_name))
            .collect())
    }

    /// Summarise the `VK_KHR_video_*` extension family in one shot.
    /// Returns an empty (all-false) struct if `extension_names()`
    /// fails — callers that care about the failure path should call
    /// `extension_names()` directly.
    pub fn supports_video_extensions(&self) -> VideoExtensionSupport {
        let names = match self.extension_names() {
            Ok(n) => n,
            Err(_) => return VideoExtensionSupport::default(),
        };
        let mut s = VideoExtensionSupport::default();
        for name in &names {
            match name.as_str() {
                VK_KHR_VIDEO_QUEUE_NAME => s.queue_khr = true,
                VK_KHR_VIDEO_DECODE_H264_NAME => s.decode_h264 = true,
                VK_KHR_VIDEO_DECODE_H265_NAME => s.decode_h265 = true,
                VK_KHR_VIDEO_DECODE_AV1_NAME => s.decode_av1 = true,
                VK_KHR_VIDEO_ENCODE_H264_NAME => s.encode_h264 = true,
                VK_KHR_VIDEO_ENCODE_H265_NAME => s.encode_h265 = true,
                _ => {}
            }
        }
        s
    }

    /// Indices of queue families that advertise either
    /// `VK_QUEUE_VIDEO_DECODE_BIT_KHR` or
    /// `VK_QUEUE_VIDEO_ENCODE_BIT_KHR`.
    ///
    /// Uses `vkGetPhysicalDeviceQueueFamilyProperties2` so we can
    /// chain `VkQueueFamilyVideoPropertiesKHR` per element — the
    /// chain is allocated here but the returned indices ignore it
    /// (Round 2 only needs the bare flag bit; the codec-operation
    /// bitmask in the chained struct is for Round 3+).
    pub fn video_queue_family_indices(&self) -> Vec<u32> {
        let mut count: u32 = 0;
        // SAFETY: count probe — null pointer for the array.
        unsafe {
            (self.fns.get_physical_device_queue_family_properties2)(
                self.handle,
                &mut count,
                ptr::null_mut(),
            );
        }

        if count == 0 {
            return Vec::new();
        }

        // Allocate `count` `VkQueueFamilyProperties2` records and
        // matching `VkQueueFamilyVideoPropertiesKHR` extension
        // structs to chain off each `pNext`.
        let mut video_props: Vec<VkQueueFamilyVideoPropertiesKHR> = (0..count)
            .map(|_| VkQueueFamilyVideoPropertiesKHR {
                s_type: VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR,
                p_next: ptr::null_mut(),
                video_codec_operations: 0,
            })
            .collect();

        let mut props: Vec<VkQueueFamilyProperties2> = (0..count as usize)
            .map(|i| VkQueueFamilyProperties2 {
                s_type: VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2,
                // SAFETY: every `video_props[i]` outlives this
                // function-scope vector for the duration of the
                // call. We don't read the chained struct after
                // the API returns (Round 2 only uses the queue
                // flag bits) but the chain has to be valid for
                // the call.
                p_next: &mut video_props[i] as *mut _ as *mut c_void,
                queue_family_properties: crate::sys::VkQueueFamilyProperties {
                    queue_flags: 0,
                    queue_count: 0,
                    timestamp_valid_bits: 0,
                    min_image_transfer_granularity: crate::sys::VkExtent3D {
                        width: 0,
                        height: 0,
                        depth: 0,
                    },
                },
            })
            .collect();

        // SAFETY: `props` is sized to the count returned by the
        // probe; the `pNext` chain is non-null and points at a
        // matching `_VIDEO_PROPERTIES_KHR` struct in the parallel
        // `video_props` vector.
        unsafe {
            (self.fns.get_physical_device_queue_family_properties2)(
                self.handle,
                &mut count,
                props.as_mut_ptr(),
            );
        }

        props
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let flags = p.queue_family_properties.queue_flags;
                if flags & (VK_QUEUE_VIDEO_DECODE_BIT_KHR | VK_QUEUE_VIDEO_ENCODE_BIT_KHR) != 0 {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .collect()
    }
}

// ─────────────────────────── helpers ─────────────────────────────────────────

/// Decode a NUL-terminated `c_char` array (Vulkan's fixed-size string
/// fields) into an owned `String`. Lossy on invalid UTF-8, which
/// shouldn't happen for any spec-conforming driver but the safe
/// thing is to not panic on a malformed driver.
fn c_chars_to_string(buf: &[std::os::raw::c_char]) -> String {
    // Find the first NUL.
    let nul = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    // SAFETY: bytes 0..nul are guaranteed not to contain a NUL by
    // construction; the buffer storage itself is at least `nul + 1`
    // bytes if we found a NUL, or `buf.len()` if we didn't.
    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, nul) };
    String::from_utf8_lossy(bytes).into_owned()
}
