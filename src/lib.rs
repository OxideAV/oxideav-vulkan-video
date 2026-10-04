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
//! # Importing an existing device
//!
//! Applications that already own a Vulkan device (a renderer, a
//! compute pipeline, …) can run decode on it instead of letting the
//! crate create its own instance / device:
//!
//! * [`device::ExternalDevice`] bundles the raw
//!   `VkInstance` / `VkPhysicalDevice` / `VkDevice` handles plus the
//!   video queue (family, index) to submit on.
//! * `decoder::H264VkDecoder::make_with_device` (behind the
//!   default-on `registry` feature) builds a framework
//!   `oxideav_core::Decoder` on those handles.
//! * The lower-level non-owning wrappers — [`Instance::from_raw`],
//!   [`Instance::from_raw_with_loader`],
//!   [`Instance::physical_device_from_raw`], [`Device::from_raw`] —
//!   are available for tooling that wants the raw bridge only.
//!
//! Imported handles are never destroyed by this crate; `Drop` only
//! tears down the objects the crate created on top of them. See the
//! safety contracts on each constructor.
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

#[cfg(feature = "registry")]
pub mod decoder;

#[cfg(feature = "registry")]
pub mod engine;

pub use device::{Device, ExternalDevice, Queue};
pub use instance::{Instance, VkError};
pub use physical_device::{PhysicalDevice, PhysicalDeviceProperties, VideoExtensionSupport};
pub use video::{
    av1_level_label, av1_profile_label, h264_level_label, h264_profile_label, h265_level_label,
    h265_profile_label, query_video_decode_av1_capabilities, query_video_decode_h264_capabilities,
    query_video_decode_h265_capabilities, VideoDecodeAV1Capabilities, VideoDecodeH264Capabilities,
    VideoDecodeH265Capabilities, VideoSession,
};

#[cfg(feature = "registry")]
pub use engine::engine_info;

/// Register Vulkan Video decode factories (Round 4).
///
/// On hosts where the Vulkan loader cannot be opened — no ICD
/// installed, headless CI without Mesa, Windows host without GPU
/// drivers, etc. — the function logs and returns without registering
/// anything; the framework's pure-Rust H.264 path remains the only
/// resolution candidate.
///
/// The H.264 decoder factory is registered at
/// [`H264_DECODE_PRIORITY`], behind the pure-Rust decoder (see
/// there). The factory itself surfaces `Error::Unsupported` if the
/// Vulkan device disagrees at runtime, so the registry will fall back
/// to the next implementation.
/// Resolution priority of the Vulkan Video H.264 decoder (lower wins).
///
/// The decoder still covers IDR pictures only (a single setup
/// reference slot, no inter prediction), so it must not outrank the
/// complete pure-Rust `h264_sw` (priority 100): it used to win
/// selection on any Vulkan-Video-capable host and real streams decoded
/// to a single frame. It stays registered so callers can opt in with
/// `CodecPreferences::prefer` / `require_hardware`.
pub const H264_DECODE_PRIORITY: i32 = 150;

#[cfg(feature = "registry")]
pub fn register(ctx: &mut oxideav_core::RuntimeContext) {
    use oxideav_core::{CodecCapabilities, CodecId, CodecInfo, CodecTag};

    match sys::framework() {
        Ok(_) => {}
        Err(e) => {
            eprintln!("oxideav-vulkan-video: library unavailable, skipping registration: {e}");
            return;
        }
    }

    let h264_caps = CodecCapabilities::video("h264_vulkan")
        .with_lossy(true)
        .with_intra_only(false)
        .with_hardware(true)
        .with_priority(H264_DECODE_PRIORITY);

    ctx.codecs.register(
        CodecInfo::new(CodecId::new("h264"))
            .capabilities(h264_caps.with_decode())
            .decoder(decoder::H264VkDecoder::make)
            .tags([
                CodecTag::fourcc(b"H264"),
                CodecTag::fourcc(b"h264"),
                CodecTag::fourcc(b"AVC1"),
                CodecTag::fourcc(b"avc1"),
                CodecTag::fourcc(b"X264"),
                CodecTag::matroska("V_MPEG4/ISO/AVC"),
            ])
            .with_engine_id("vulkan-video")
            .with_engine_probe(engine::engine_info),
    );
}

#[cfg(feature = "registry")]
oxideav_core::register!("vulkan-video", register);
