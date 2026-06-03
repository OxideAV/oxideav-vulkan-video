//! Vulkan Video engine probe — `engine_info()` enumerates every
//! Vulkan physical device the loader can see and surfaces the
//! per-codec video capabilities used by `oxideav-cli`'s `info`
//! command.
//!
//! The function is a `cfg(any(target_os = "linux", target_os =
//! "windows"))` — the rest of the crate is too. On platforms where
//! the entire crate compiles to an empty rlib (macOS today),
//! consumers don't see this module.
//!
//! Discovery model:
//!
//! 1. [`crate::Instance::new`] is called with API version 1.2 — the
//!    same target the rest of the crate's tests use. Failure
//!    (loader missing, no ICD installed, headless CI) collapses to
//!    `vec![]`; consumers treat the empty result as "no Vulkan
//!    backend on this host".
//! 2. For every physical device:
//!    * `properties()` gives us the human-readable name, packed
//!      `apiVersion` / `driverVersion`, vendor / device PCI ids,
//!      and the device type.
//!    * Devices that are neither `DiscreteGpu`, `IntegratedGpu`,
//!      `VirtualGpu`, nor advertise any `VK_KHR_video_*`
//!      extension are skipped — a CPU-only ICD with no video
//!      capability isn't useful and would only add noise.
//!    * The per-heap `VkPhysicalDeviceMemoryProperties` table is
//!      summed across every heap whose flags set
//!      `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT`. Discrete GPUs report
//!      their VRAM here; integrated parts typically report the
//!      shared-RAM heap they reserve.
//!    * For each codec the device advertises `VK_KHR_video_decode_*`
//!      (or `_encode_*`) for, an entry is appended to
//!      [`HwCodecCaps`]. H.264 decode goes through
//!      [`crate::query_video_decode_h264_capabilities`] for actual
//!      max-extent / DPB-slot numbers; HEVC and AV1 emit a
//!      decode-only flag without dimensions because the matching
//!      capability-query plumbing isn't wired up yet.
//!
//! The probe is idempotent and side-effect free. Consumers may call
//! it many times per process; each call opens its own
//! [`crate::Instance`] and drops it on return.

#![cfg(any(target_os = "linux", target_os = "windows"))]
#![cfg(feature = "registry")]

use oxideav_core::{HwCodecCaps, HwDeviceInfo};

use crate::physical_device::{PhysicalDevice, PhysicalDeviceType, VideoExtensionSupport};
use crate::sys::{
    vk_api_version_major, vk_api_version_minor, vk_api_version_patch, StdVideoAV1Profile,
    StdVideoH264ProfileIdc, StdVideoH265ProfileIdc, VkPhysicalDeviceMemoryProperties,
    STD_VIDEO_AV1_PROFILE_MAIN, STD_VIDEO_H264_PROFILE_IDC_HIGH, STD_VIDEO_H265_PROFILE_IDC_MAIN,
    VK_API_VERSION_1_2, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
};
use crate::video::{
    query_video_decode_av1_capabilities, query_video_decode_h264_capabilities,
    query_video_decode_h265_capabilities,
};
use crate::Instance;

/// Enumerate Vulkan Video engines on this host.
///
/// Returns one [`HwDeviceInfo`] per physical device the Vulkan loader
/// reports, populated with name, driver/API version, on-card memory,
/// and a per-codec [`HwCodecCaps`] entry for every codec that's
/// advertised through the `VK_KHR_video_*` extension family.
///
/// Skip-friendly: returns `vec![]` on any error path (no ICD, no
/// loader, broken driver). Consumers treat `is_empty()` as "no
/// Vulkan backend on this host" and fall back to whatever pure-Rust
/// path they have.
pub fn engine_info() -> Vec<HwDeviceInfo> {
    let instance = match Instance::new("oxideav-vulkan-video-engine-info", VK_API_VERSION_1_2) {
        Ok(i) => i,
        Err(_) => return Vec::new(),
    };
    let pds = match instance.physical_devices() {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut out: Vec<HwDeviceInfo> = Vec::with_capacity(pds.len());
    for pd in &pds {
        if let Some(info) = build_device_info(pd) {
            out.push(info);
        }
    }
    out
}

/// Build one [`HwDeviceInfo`] for a single physical device. Returns
/// `None` for devices that have neither a useful type
/// (`DiscreteGpu` / `IntegratedGpu` / `VirtualGpu`) nor any
/// `VK_KHR_video_*` capability — a software CPU ICD with no video
/// extension isn't worth surfacing.
fn build_device_info(pd: &PhysicalDevice<'_>) -> Option<HwDeviceInfo> {
    let props = pd.properties();
    let video = pd.supports_video_extensions();

    if !device_type_is_useful(props.device_type) && !any_video_extension(&video) {
        return None;
    }

    let api_version = format!(
        "Vulkan {}.{}.{}",
        vk_api_version_major(props.api_version),
        vk_api_version_minor(props.api_version),
        vk_api_version_patch(props.api_version),
    );

    let driver_version = format!("0x{:08x}", props.driver_version);

    let extra = vec![
        (
            "vendor_id".to_string(),
            format!("0x{:04x}", props.vendor_id),
        ),
        (
            "device_id".to_string(),
            format!("0x{:04x}", props.device_id),
        ),
        (
            "device_type".to_string(),
            device_type_label(props.device_type).to_string(),
        ),
    ];

    let total_memory_bytes = device_local_memory_bytes(pd);

    let codecs = build_codec_caps(pd, &video);

    Some(HwDeviceInfo {
        name: props.name,
        driver_version: Some(driver_version),
        api_version: Some(api_version),
        total_memory_bytes,
        extra,
        codecs,
    })
}

/// Construct the per-codec [`HwCodecCaps`] vector for this device.
///
/// H.264 caps are populated from the existing
/// [`query_video_decode_h264_capabilities`] entry — High profile
/// is queried (it's the most permissive 8-bit 4:2:0 profile every
/// modern NVDEC / AMD VCN / Intel Quick Sync exposes). HEVC and AV1
/// are advertised as decode-capable when the matching
/// `VK_KHR_video_decode_h265` / `VK_KHR_video_decode_av1` extension
/// is present, but without a dimension query — that plumbing is a
/// follow-up round.
fn build_codec_caps(pd: &PhysicalDevice<'_>, video: &VideoExtensionSupport) -> Vec<HwCodecCaps> {
    let mut codecs = Vec::with_capacity(3);

    if video.decode_h264 || video.encode_h264 {
        codecs.push(build_h264_caps(pd, video));
    }
    if video.decode_h265 || video.encode_h265 {
        codecs.push(build_hevc_caps(pd, video));
    }
    if video.decode_av1 {
        codecs.push(build_av1_caps(pd));
    }

    codecs
}

/// H.264: query the actual decode capabilities for the High profile
/// when decode is advertised. Encode is reported as a flag based on
/// the extension list — Round 6 doesn't query the encode caps chain
/// (no `VkVideoEncodeH264CapabilitiesKHR` plumbing yet).
fn build_h264_caps(pd: &PhysicalDevice<'_>, video: &VideoExtensionSupport) -> HwCodecCaps {
    let mut caps = HwCodecCaps {
        codec: "h264".to_string(),
        decode: video.decode_h264,
        encode: video.encode_h264,
        max_width: None,
        max_height: None,
        max_bit_depth: Some(8),
        profiles: Vec::new(),
        extra: Vec::new(),
    };

    if video.decode_h264 {
        // Query against the H.264 High profile — every modern
        // implementation that exposes decode_h264 supports it, and
        // it's the most permissive of the standard profiles.
        let profile: StdVideoH264ProfileIdc = STD_VIDEO_H264_PROFILE_IDC_HIGH;
        match query_video_decode_h264_capabilities(pd, profile) {
            Ok(h264) => {
                caps.max_width = Some(h264.max_coded_extent.0);
                caps.max_height = Some(h264.max_coded_extent.1);
                caps.profiles = vec!["High".to_string()];
                caps.extra
                    .push(("max_dpb_slots".to_string(), h264.max_dpb_slots.to_string()));
                caps.extra.push((
                    "max_active_reference_pictures".to_string(),
                    h264.max_active_reference_pictures.to_string(),
                ));
                caps.extra
                    .push(("max_level_idc".to_string(), h264.max_level_idc.to_string()));
                let std_header = extension_name_string(&h264.std_header_version.extension_name);
                if !std_header.is_empty() {
                    caps.extra.push(("std_header".to_string(), std_header));
                }
                caps.extra.push((
                    "std_header_version".to_string(),
                    format_video_std_version(h264.std_header_version.spec_version),
                ));
            }
            Err(_) => {
                // Capability query failed — keep the decode flag
                // (the extension is advertised) but leave dimensions
                // unset.
            }
        }
    }

    caps
}

/// HEVC: query the actual decode capabilities for the H.265 Main
/// profile when decode is advertised. Main is the universally-
/// supported 8-bit 4:2:0 profile — Main 10 / Range Extensions are a
/// follow-up (they need different bit-depth flags in the chained
/// profile struct). Encode is reported as a flag based on the
/// extension list; the matching `VkVideoEncodeH265CapabilitiesKHR`
/// plumbing isn't wired up.
fn build_hevc_caps(pd: &PhysicalDevice<'_>, video: &VideoExtensionSupport) -> HwCodecCaps {
    let mut caps = HwCodecCaps {
        codec: "hevc".to_string(),
        decode: video.decode_h265,
        encode: video.encode_h265,
        max_width: None,
        max_height: None,
        max_bit_depth: Some(8),
        profiles: Vec::new(),
        extra: Vec::new(),
    };

    if video.decode_h265 {
        let profile: StdVideoH265ProfileIdc = STD_VIDEO_H265_PROFILE_IDC_MAIN;
        if let Ok(h265) = query_video_decode_h265_capabilities(pd, profile) {
            caps.max_width = Some(h265.max_coded_extent.0);
            caps.max_height = Some(h265.max_coded_extent.1);
            caps.profiles = vec!["Main".to_string()];
            caps.extra
                .push(("max_dpb_slots".to_string(), h265.max_dpb_slots.to_string()));
            caps.extra.push((
                "max_active_reference_pictures".to_string(),
                h265.max_active_reference_pictures.to_string(),
            ));
            caps.extra
                .push(("max_level_idc".to_string(), h265.max_level_idc.to_string()));
            let std_header = extension_name_string(&h265.std_header_version.extension_name);
            if !std_header.is_empty() {
                caps.extra.push(("std_header".to_string(), std_header));
            }
            caps.extra.push((
                "std_header_version".to_string(),
                format_video_std_version(h265.std_header_version.spec_version),
            ));
        }
    }

    caps
}

/// AV1: query the actual decode capabilities for the Main profile
/// (8-bit / 10-bit 4:2:0, no film-grain commitment). Vulkan Video
/// does not standardise an `_encode_av1` extension; encode is
/// hard-coded `false` here.
fn build_av1_caps(pd: &PhysicalDevice<'_>) -> HwCodecCaps {
    let mut caps = HwCodecCaps {
        codec: "av1".to_string(),
        decode: true,
        encode: false,
        max_width: None,
        max_height: None,
        max_bit_depth: Some(8),
        profiles: Vec::new(),
        extra: Vec::new(),
    };

    let profile: StdVideoAV1Profile = STD_VIDEO_AV1_PROFILE_MAIN;
    if let Ok(av1) = query_video_decode_av1_capabilities(pd, profile, false) {
        caps.max_width = Some(av1.max_coded_extent.0);
        caps.max_height = Some(av1.max_coded_extent.1);
        caps.profiles = vec!["Main".to_string()];
        caps.extra
            .push(("max_dpb_slots".to_string(), av1.max_dpb_slots.to_string()));
        caps.extra.push((
            "max_active_reference_pictures".to_string(),
            av1.max_active_reference_pictures.to_string(),
        ));
        caps.extra
            .push(("max_level".to_string(), av1.max_level.to_string()));
        let std_header = extension_name_string(&av1.std_header_version.extension_name);
        if !std_header.is_empty() {
            caps.extra.push(("std_header".to_string(), std_header));
        }
        caps.extra.push((
            "std_header_version".to_string(),
            format_video_std_version(av1.std_header_version.spec_version),
        ));
    }

    caps
}

/// `true` for device types that are worth surfacing to a consumer
/// (discrete / integrated / virtual GPU). CPU and Other types are
/// only kept when they advertise a `VK_KHR_video_*` extension.
fn device_type_is_useful(t: PhysicalDeviceType) -> bool {
    matches!(
        t,
        PhysicalDeviceType::DiscreteGpu
            | PhysicalDeviceType::IntegratedGpu
            | PhysicalDeviceType::VirtualGpu
    )
}

/// Lowercase short label for `device_type` printed into the
/// `HwDeviceInfo::extra` map.
fn device_type_label(t: PhysicalDeviceType) -> &'static str {
    match t {
        PhysicalDeviceType::Other => "other",
        PhysicalDeviceType::IntegratedGpu => "integrated",
        PhysicalDeviceType::DiscreteGpu => "discrete",
        PhysicalDeviceType::VirtualGpu => "virtual",
        PhysicalDeviceType::Cpu => "cpu",
    }
}

/// Whether at least one `VK_KHR_video_*` extension is advertised by
/// the device.
fn any_video_extension(v: &VideoExtensionSupport) -> bool {
    v.queue_khr || v.decode_h264 || v.decode_h265 || v.decode_av1 || v.encode_h264 || v.encode_h265
}

/// Sum of every heap on the physical device whose `flags` field has
/// `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT` set. Returns `None` on a query
/// path that emits no device-local heap (rare; defensive).
fn device_local_memory_bytes(pd: &PhysicalDevice<'_>) -> Option<u64> {
    // SAFETY: zero-initialised buffer of the right size+layout for
    // VkPhysicalDeviceMemoryProperties; Vulkan writes every populated
    // field.
    let mut props: VkPhysicalDeviceMemoryProperties = unsafe { std::mem::zeroed() };
    // SAFETY: `props` is a valid local of the right type; the
    // physical-device handle is owned by the parent instance and
    // hasn't been destroyed.
    unsafe {
        (pd.instance_fns().get_physical_device_memory_properties)(pd.handle(), &mut props);
    }
    let count = props.memory_heap_count as usize;
    let total: u64 = props
        .memory_heaps
        .iter()
        .take(count)
        .filter(|h| h.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0)
        .map(|h| h.size)
        .sum();

    if total == 0 {
        None
    } else {
        Some(total)
    }
}

/// Decode the NUL-terminated `extensionName` field of a
/// `VkExtensionProperties` into an owned `String`. Empty when the
/// buffer is all zeroes.
fn extension_name_string(buf: &[std::os::raw::c_char]) -> String {
    let nul = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    // SAFETY: bytes 0..nul are non-NUL by construction.
    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, nul) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// Decode `VK_MAKE_VIDEO_STD_VERSION(major, minor, patch)` packing —
/// `major << 22 | minor << 12 | patch`. Inverse of the constant
/// definition for [`crate::sys::VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_SPEC_VERSION`].
fn format_video_std_version(packed: u32) -> String {
    let major = (packed >> 22) & 0x3FF;
    let minor = (packed >> 12) & 0x3FF;
    let patch = packed & 0xFFF;
    format!("{major}.{minor}.{patch}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sys;

    #[test]
    fn engine_info_returns_or_empties() {
        // Exercise the public API. On any host without a Vulkan ICD
        // we get vec![]; on the dev box we get one or more entries.
        // Either way the function must not panic.
        let _ = engine_info();
    }

    #[test]
    fn format_video_std_version_round_trips_known_packing() {
        // Spec version constant is `1.0.0` packed.
        assert_eq!(
            format_video_std_version(sys::VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_SPEC_VERSION),
            "1.0.0",
        );
        assert_eq!(format_video_std_version((3 << 22) | (2 << 12) | 1), "3.2.1");
    }

    #[test]
    fn extension_name_string_handles_empty_and_utf8() {
        use std::os::raw::c_char;
        let buf: [c_char; 16] = [0; 16];
        assert_eq!(extension_name_string(&buf), "");
        let mut buf: [c_char; 16] = [0; 16];
        for (i, b) in b"hello".iter().enumerate() {
            buf[i] = *b as c_char;
        }
        assert_eq!(extension_name_string(&buf), "hello");
    }

    #[test]
    fn device_type_label_covers_every_variant() {
        assert_eq!(device_type_label(PhysicalDeviceType::Other), "other");
        assert_eq!(
            device_type_label(PhysicalDeviceType::IntegratedGpu),
            "integrated"
        );
        assert_eq!(
            device_type_label(PhysicalDeviceType::DiscreteGpu),
            "discrete"
        );
        assert_eq!(device_type_label(PhysicalDeviceType::VirtualGpu), "virtual");
        assert_eq!(device_type_label(PhysicalDeviceType::Cpu), "cpu");
    }
}
