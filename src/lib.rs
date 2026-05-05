#![cfg(target_os = "linux")]
//! Linux Vulkan Video hardware decode/encode bridge.
//!
//! This crate is a **runtime-loaded** bridge to the Vulkan loader
//! (`libvulkan.so.1`). It uses [`libloading`] to dlopen the loader on
//! first use, so:
//!
//! * Linux builds have **no compile-time link dependency** on Vulkan;
//!   if the loader can't be loaded (no Vulkan ICD installed, headless
//!   CI without Mesa, etc.) the registered factories return
//!   `Error::Unsupported` and the framework registry falls back to
//!   the pure-Rust codec implementation.
//! * No bindgen, no `*-sys` crate. Vulkan is a C API; symbol
//!   resolution and `VkResult` propagation is all done by hand.
//!
//! The crate is gated to `cfg(target_os = "linux")` at the source
//! level: on macOS / Windows the entire crate compiles to an empty
//! rlib, and consumers (umbrella `oxideav`) gate the `register` call
//! behind the same cfg.
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
//! Round 1 (this commit): scaffolding only. The framework load is
//! verified via `sys::framework()`; no codec factories are wired up
//! yet. Round 2 will create a `VkInstance`, query the
//! `VK_KHR_video_*` extension family, and add H.264 + HEVC decode.
//!
//! # Workspace policy
//!
//! Calling a system OS / driver API via FFI is the same shape as
//! calling `libc::malloc` — it's the platform, not a copied
//! algorithm. The workspace's clean-room rule (no embedding source
//! from libvpx, libwebp, libjxl, etc.) doesn't apply here.

pub mod sys;

/// Confirm the Vulkan loader loads, but do not register any codec
/// factories yet (Round 1 scaffolding).
///
/// If `libvulkan.so.1` cannot be loaded (no Vulkan ICD, headless CI
/// without Mesa, etc.) the function logs and returns — the runtime
/// falls back to the pure-Rust impls.
#[cfg(feature = "registry")]
pub fn register(_ctx: &mut oxideav_core::RuntimeContext) {
    match sys::framework() {
        Ok(_) => {
            // Round 1: framework loads. No factories wired up yet.
        }
        Err(e) => {
            eprintln!("oxideav-vulkan-video: library unavailable, skipping registration: {e}");
        }
    }
}

#[cfg(feature = "registry")]
oxideav_core::register!("vulkan-video", register);
