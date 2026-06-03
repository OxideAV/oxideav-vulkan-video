//! Round 8 integration tests — H.265 (HEVC) and AV1 capability
//! queries.
//!
//! Same skip-on-no-Vulkan policy as the earlier round-N tests: each
//! test prints a skip notice and returns rather than panicking when
//! no Vulkan ICD is reachable. On hosts where the loader is present
//! but the device doesn't advertise the matching `VK_KHR_video_*`
//! extension, the test skips that codec — Round 8 lands the query
//! plumbing, not a hardware requirement.

#![cfg(any(target_os = "linux", target_os = "windows"))]

use oxideav_vulkan_video::{
    query_video_decode_av1_capabilities, query_video_decode_h265_capabilities,
    sys::{
        STD_VIDEO_AV1_PROFILE_MAIN, STD_VIDEO_H265_LEVEL_IDC_5_1, STD_VIDEO_H265_PROFILE_IDC_MAIN,
        VK_API_VERSION_1_2,
    },
    Instance, PhysicalDevice,
};

fn try_init_instance() -> Option<Instance> {
    match Instance::new("oxideav-vulkan-video-round8", VK_API_VERSION_1_2) {
        Ok(i) => Some(i),
        Err(e) => {
            eprintln!("oxideav-vulkan-video: skipping round 8 (no Vulkan ICD): {e}");
            None
        }
    }
}

/// Find the first physical device that advertises HEVC decode.
fn pick_hevc_capable<'i>(devices: &[PhysicalDevice<'i>]) -> Option<usize> {
    devices
        .iter()
        .enumerate()
        .find(|(_, d)| d.supports_video_extensions().decode_h265)
        .map(|(i, _)| i)
}

/// Find the first physical device that advertises AV1 decode.
fn pick_av1_capable<'i>(devices: &[PhysicalDevice<'i>]) -> Option<usize> {
    devices
        .iter()
        .enumerate()
        .find(|(_, d)| d.supports_video_extensions().decode_av1)
        .map(|(i, _)| i)
}

#[test]
fn query_h265_decode_caps_main_profile() {
    let Some(inst) = try_init_instance() else {
        return;
    };
    let devices = inst.physical_devices().expect("physical_devices");
    let Some(i) = pick_hevc_capable(&devices) else {
        eprintln!("no H.265-decode-capable physical device; skipping");
        return;
    };
    let pd = &devices[i];

    let caps = query_video_decode_h265_capabilities(pd, STD_VIDEO_H265_PROFILE_IDC_MAIN)
        .expect("vkGetPhysicalDeviceVideoCapabilitiesKHR for H.265 Main");

    eprintln!("H.265 Main caps: {:?}", caps);

    assert!(
        caps.max_coded_extent.0 >= 1920 && caps.max_coded_extent.1 >= 1080,
        "expected at least HD support: got {:?}",
        caps.max_coded_extent
    );
    assert!(
        caps.max_level_idc >= STD_VIDEO_H265_LEVEL_IDC_5_1,
        "expected at least Level 5.1: got max_level_idc={}",
        caps.max_level_idc
    );
    assert!(
        caps.max_dpb_slots >= 1,
        "expected >=1 DPB slot: got {}",
        caps.max_dpb_slots
    );
}

#[test]
fn query_av1_decode_caps_main_profile() {
    let Some(inst) = try_init_instance() else {
        return;
    };
    let devices = inst.physical_devices().expect("physical_devices");
    let Some(i) = pick_av1_capable(&devices) else {
        eprintln!("no AV1-decode-capable physical device; skipping");
        return;
    };
    let pd = &devices[i];

    let caps = query_video_decode_av1_capabilities(pd, STD_VIDEO_AV1_PROFILE_MAIN, false)
        .expect("vkGetPhysicalDeviceVideoCapabilitiesKHR for AV1 Main");

    eprintln!("AV1 Main caps: {:?}", caps);

    assert!(
        caps.max_coded_extent.0 >= 1920 && caps.max_coded_extent.1 >= 1080,
        "expected at least HD support: got {:?}",
        caps.max_coded_extent
    );
    assert!(
        caps.max_dpb_slots >= 1,
        "expected >=1 DPB slot: got {}",
        caps.max_dpb_slots
    );
    // AV1 max_level is an enum index; STD_VIDEO_AV1_LEVEL_2_0=0 is the
    // smallest valid value. Anything above 0 means the device returned
    // a real level (and 0 is itself still legal — just the 2.0
    // baseline). Sanity-check it's in the valid 0..=23 range rather
    // than a poison sentinel.
    assert!(
        (0..=23).contains(&caps.max_level),
        "max_level out of expected range: got {}",
        caps.max_level
    );
}
