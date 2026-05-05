//! Round 2 integration tests.
//!
//! Same `cfg` gate as the rest of the crate: only Linux + Windows
//! compile a real test body; macOS gets an empty file.
//!
//! Failure mode: if there is no Vulkan loader / ICD on the host
//! (headless CI without Mesa, e.g.), `Instance::new` will fail and
//! the test prints a skip notice and returns rather than panicking.
//! On the dev box we expect every test to pass — the NVIDIA RTX 5080
//! advertises decode_h264 / decode_h265 / decode_av1.

#![cfg(any(target_os = "linux", target_os = "windows"))]

use oxideav_vulkan_video::{
    sys::{self, vk_api_version_major, vk_api_version_minor, VK_API_VERSION_1_2},
    Instance,
};

fn try_init_instance() -> Option<Instance> {
    match Instance::new("oxideav-vulkan-video-test", VK_API_VERSION_1_2) {
        Ok(i) => Some(i),
        Err(e) => {
            eprintln!(
                "oxideav-vulkan-video: skipping (no Vulkan ICD on this host?): {e}"
            );
            None
        }
    }
}

#[test]
fn loader_loads() {
    // Mirrors the `sys::tests::frameworks_load` smoke test, but
    // run from outside the `cfg(test)` module so the integration
    // surface is exercised too.
    sys::framework().expect("Vulkan loader must be available on the dev box");
}

#[test]
fn instance_creates_with_oxideav_app_name() {
    let Some(_inst) = try_init_instance() else {
        return;
    };
    // Drop runs vkDestroyInstance — no further assertion needed
    // here beyond the constructor having returned Ok.
}

#[test]
fn lists_at_least_one_physical_device() {
    let Some(inst) = try_init_instance() else {
        return;
    };
    let devices = inst
        .physical_devices()
        .expect("vkEnumeratePhysicalDevices");
    assert!(
        !devices.is_empty(),
        "no Vulkan physical devices reported on a host with a Vulkan ICD"
    );
}

#[test]
fn physical_device_reports_name_and_vendor() {
    let Some(inst) = try_init_instance() else {
        return;
    };
    let devices = inst
        .physical_devices()
        .expect("vkEnumeratePhysicalDevices");
    for d in &devices {
        let props = d.properties();
        assert!(
            !props.name.trim().is_empty(),
            "physical device reported empty deviceName"
        );
        eprintln!(
            "vk device: {} (vendor=0x{:04x} device=0x{:04x} api={}.{} type={:?})",
            props.name,
            props.vendor_id,
            props.device_id,
            vk_api_version_major(props.api_version),
            vk_api_version_minor(props.api_version),
            props.device_type,
        );
    }
}

#[test]
fn nvidia_advertises_video_decode_h264() {
    let Some(inst) = try_init_instance() else {
        return;
    };
    let devices = inst
        .physical_devices()
        .expect("vkEnumeratePhysicalDevices");
    let mut saw_nvidia = false;
    for d in &devices {
        let props = d.properties();
        // PCI vendor id 0x10DE = NVIDIA Corporation.
        if props.vendor_id == 0x10DE {
            saw_nvidia = true;
            let support = d.supports_video_extensions();
            eprintln!("NVIDIA device {}: {:?}", props.name, support);
            assert!(
                support.decode_h264,
                "NVIDIA Vulkan ICD must advertise VK_KHR_video_decode_h264 \
                 (Ada / Ampere / Blackwell all expose it as of driver 535+)"
            );
        }
    }
    if !saw_nvidia {
        eprintln!("no NVIDIA GPU detected; skipping decode_h264 assertion");
    }
}

#[test]
fn video_queue_family_indices_smoke() {
    let Some(inst) = try_init_instance() else {
        return;
    };
    let devices = inst
        .physical_devices()
        .expect("vkEnumeratePhysicalDevices");
    for d in &devices {
        let props = d.properties();
        let indices = d.video_queue_family_indices();
        eprintln!(
            "{}: {} video-capable queue families (indices={:?})",
            props.name,
            indices.len(),
            indices
        );
        // No assertion: drivers without VK_KHR_video_queue
        // legitimately return an empty list.
    }
}
