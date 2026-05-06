//! Round 6 integration test for `oxideav_vulkan_video::engine_info`.
//!
//! Same `cfg` gate as the rest of the crate. Skip-friendly when no
//! Vulkan ICD is installed (headless CI without Mesa, e.g.) — the
//! function returns `vec![]` and the test logs and exits without
//! failing.

#![cfg(any(target_os = "linux", target_os = "windows"))]
#![cfg(feature = "registry")]

#[test]
fn engine_info_finds_rtx_5080_or_skips() {
    let devs = oxideav_vulkan_video::engine_info();
    if devs.is_empty() {
        eprintln!("No Vulkan ICD — skip");
        return;
    }
    eprintln!("found {} Vulkan device(s)", devs.len());
    let dev = &devs[0];
    eprintln!("device 0: {:?}", dev);
    assert!(!dev.name.is_empty());
    assert!(dev.api_version.is_some());
    let h264 = dev.codecs.iter().find(|c| c.codec == "h264");
    assert!(h264.is_some(), "h264 caps entry");
}

#[test]
fn engine_info_per_device_metadata_is_consistent() {
    // Every entry must have non-empty name, api_version Some, and the
    // standard `vendor_id` / `device_id` / `device_type` extras keyed
    // exactly so the CLI can rely on them.
    let devs = oxideav_vulkan_video::engine_info();
    if devs.is_empty() {
        eprintln!("No Vulkan ICD — skip");
        return;
    }
    for d in &devs {
        assert!(!d.name.is_empty(), "device name must not be empty");
        assert!(d.api_version.is_some(), "api_version must be reported");
        assert!(
            d.driver_version.is_some(),
            "driver_version must be reported"
        );
        let keys: Vec<_> = d.extra.iter().map(|(k, _)| k.as_str()).collect();
        assert!(
            keys.contains(&"vendor_id"),
            "extras should expose vendor_id, got {keys:?}",
        );
        assert!(
            keys.contains(&"device_id"),
            "extras should expose device_id, got {keys:?}",
        );
        assert!(
            keys.contains(&"device_type"),
            "extras should expose device_type, got {keys:?}",
        );
        // Every codec entry's name should be a non-empty lowercase
        // string matching one of the known codec ids.
        for c in &d.codecs {
            assert!(!c.codec.is_empty());
            assert!(matches!(c.codec.as_str(), "h264" | "hevc" | "av1"));
        }
    }
}

#[test]
fn engine_info_attaches_via_with_engine_probe() {
    // Round-trip the function through the `EngineProbeFn` typedef
    // exposed by oxideav-core to make sure it satisfies the
    // signature contract (i.e. `fn() -> Vec<HwDeviceInfo>` — no
    // closures, no `impl Fn`).
    let probe: oxideav_core::EngineProbeFn = oxideav_vulkan_video::engine_info;
    let _ = probe();
}
