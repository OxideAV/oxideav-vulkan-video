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
//! Round 2 (this commit): the bootstrap → `VkInstance` →
//! `VkPhysicalDevice` → `VK_KHR_video_*` extension probe path is
//! plumbed end-to-end via the safe wrappers in
//! [`instance::Instance`] and [`physical_device::PhysicalDevice`].
//! `register()` remains a graceful no-op — no codec factories are
//! wired up yet. Round 3 will add the first decode session
//! (H.264 / HEVC) layered on top.
//!
//! # Workspace policy
//!
//! Calling a system OS / driver API via FFI is the same shape as
//! calling `libc::malloc` — it's the platform, not a copied
//! algorithm. The workspace's clean-room rule (no embedding source
//! from libvpx, libwebp, libjxl, etc.) doesn't apply here.

pub mod instance;
pub mod physical_device;
pub mod sys;

pub use instance::{Instance, VkError};
pub use physical_device::{PhysicalDevice, PhysicalDeviceProperties, VideoExtensionSupport};

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
