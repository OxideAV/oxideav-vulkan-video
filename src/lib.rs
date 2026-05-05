#![cfg(any(target_os = "linux", target_os = "windows"))]
//! Vulkan Video hardware decode/encode bridge (Linux + Windows).
//!
//! This crate is a **runtime-loaded** bridge to the Vulkan loader:
//! `libvulkan.so.1` on Linux, `vulkan-1.dll` on Windows. It uses
//! [`libloading`] to dlopen / `LoadLibrary` the loader on first use,
//! so:
//!
//! * Builds have **no compile-time link dependency** on Vulkan; if
//!   the loader can't be loaded (no Vulkan ICD installed, headless
//!   CI without Mesa, Windows host without GPU driver, etc.) the
//!   registered factories return `Error::Unsupported` and the
//!   framework registry falls back to the pure-Rust codec
//!   implementation.
//! * No bindgen, no `*-sys` crate. Vulkan is a C API; symbol
//!   resolution and `VkResult` propagation is all done by hand.
//!
//! The crate is gated to `cfg(any(target_os = "linux", target_os =
//! "windows"))` at the source level: on macOS the entire crate
//! compiles to an empty rlib (Vulkan is reachable via MoltenVK there
//! but with a different loading story; out of scope for now), and
//! consumers (umbrella `oxideav`) gate the `register` call behind
//! the same cfg.
//!
//! # Programming model
//!
//! Vulkan exposes only four entries as ordinary dlsym targets —
//! `vkGetInstanceProcAddr`, `vkCreateInstance`,
//! `vkEnumerateInstanceExtensionProperties`,
//! `vkEnumerateInstanceVersion`. Every other Vulkan function,
//! including all `VK_KHR_video_*` entry points, is reached via
//! `vkGetInstanceProcAddr` (instance-level) or `vkGetDeviceProcAddr`
//! (device-level) after a `VkInstance` is constructed. So the
//! bootstrap vtable is intentionally tiny; Round 2 will populate the
//! post-instance dispatch surface.
//!
//! # Status
//!
//! Round 3 (this commit): adds [`device::Device`] (a logical
//! `VkDevice` opened against a video-decode queue family) and
//! [`video::VideoSession`] (a `VkVideoSessionKHR` with backing
//! `VkDeviceMemory` bound through `vkBindVideoSessionMemoryKHR`).
//! Capability queries for H.264 decode profiles are wired up via
//! [`video::query_video_decode_h264_capabilities`]. `register()`
//! remains a graceful no-op — actual decode-loop submission
//! (`vkCmdBeginVideoCodingKHR` / `vkCmdDecodeVideoKHR` /
//! `vkCmdEndVideoCodingKHR`) is Round 4 territory.
//!
//! # Workspace policy
//!
//! Calling a system OS / driver API via FFI is the same shape as
//! calling `libc::malloc` — it's the platform, not a copied
//! algorithm. The workspace's clean-room rule (no embedding source
//! from libvpx, libwebp, libjxl, etc.) doesn't apply here.

pub mod device;
pub mod instance;
pub mod physical_device;
pub mod sys;
pub mod video;

pub use device::{Device, Queue};
pub use instance::{Instance, VkError};
pub use physical_device::{PhysicalDevice, PhysicalDeviceProperties, VideoExtensionSupport};
pub use video::{query_video_decode_h264_capabilities, VideoDecodeH264Capabilities, VideoSession};

/// Confirm the Vulkan loader loads, but do not register any codec
/// factories yet (Round 2 still defers registration).
///
/// If `libvulkan.so.1` cannot be loaded (no Vulkan ICD, headless CI
/// without Mesa, etc.) the function logs and returns — the runtime
/// falls back to the pure-Rust impls.
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut oxideav_core::RuntimeContext) {
    match sys::framework() {
        Ok(_) => {
            // Round 2: framework loads + safe instance/physical-device
            // wrappers exist; no codec factories yet.
        }
        Err(e) => {
            eprintln!("oxideav-vulkan-video: library unavailable, skipping registration: {e}");
        }
    }
}

#[cfg(feature = "registry")]
oxideav_core::register!("vulkan-video", register);
