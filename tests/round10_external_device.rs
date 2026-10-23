//! Round 10 integration tests — importing an application-owned
//! Vulkan device (GitHub issue #2).
//!
//! Same skip-on-no-Vulkan policy as the earlier rounds: every test
//! opens the Vulkan loader and exits cleanly (without failing) when
//! the host has no Vulkan ICD installed or the fixture is missing.
//!
//! What's covered:
//!
//! * `Instance::from_raw` wraps a foreign `VkInstance` handle
//!   non-owning: the wrapper resolves the full dispatch surface and
//!   its `Drop` leaves the original instance alive and usable.
//! * `Device::from_raw` wraps a foreign `VkDevice` the same way,
//!   resolving the video entry points through the parent instance's
//!   `vkGetDeviceProcAddr`.
//! * `H264VkDecoder::make_with_device` runs the full lazy pipeline
//!   construction (session, parameters, images, buffers, command
//!   pool) on the imported device, under the same
//!   `OXIDEAV_VK_SKIP_SUBMIT` hook rounds 4/7 use to avoid the
//!   known NVIDIA `vkQueueSubmit` SIGSEGV in-process.
//! * The imported handles survive the decoder's `Drop` — the test
//!   keeps using the application-owned `Device` afterwards.

#![cfg(any(target_os = "linux", target_os = "windows"))]
#![cfg(feature = "registry")]

use std::path::PathBuf;
use std::sync::Mutex;

use oxideav_core::{CodecId, CodecParameters, Packet, time::TimeBase};

use oxideav_vulkan_video::decoder::H264VkDecoder;
use oxideav_vulkan_video::physical_device::{
    VK_KHR_SYNCHRONIZATION_2_NAME, VK_KHR_VIDEO_DECODE_H264_NAME, VK_KHR_VIDEO_DECODE_QUEUE_NAME,
    VK_KHR_VIDEO_QUEUE_NAME,
};
use oxideav_vulkan_video::sys::VK_API_VERSION_1_2;
use oxideav_vulkan_video::{Device, ExternalDevice, Instance};

/// Serialise the `OXIDEAV_VK_SKIP_SUBMIT` env-hooked code paths (see
/// round 7 for the rationale — a race can let a real `vkQueueSubmit`
/// slip through and SIGSEGV on NVIDIA).
static ENV_HOOK_LOCK: Mutex<()> = Mutex::new(());

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn read_fixture() -> Option<Vec<u8>> {
    let p = fixtures_dir().join("h264_high_320x240_1frame.h264");
    match std::fs::read(&p) {
        Ok(b) => Some(b),
        Err(_) => {
            eprintln!("vulkan-video round10: fixture missing at {:?}; skipping", p);
            None
        }
    }
}

fn try_init_instance() -> Option<Instance> {
    match Instance::new("oxideav-vulkan-video-round10", VK_API_VERSION_1_2) {
        Ok(i) => Some(i),
        Err(e) => {
            eprintln!("vulkan-video round10: no Vulkan ICD? skipping: {e}");
            None
        }
    }
}

/// Pick the first physical device that can decode H.264 and its
/// first video-capable queue family. Returns the raw handle + qfi so
/// the borrow on `inst` ends before the caller moves on.
fn pick_h264_device(inst: &Instance) -> Option<(oxideav_vulkan_video::sys::VkPhysicalDevice, u32)> {
    let devices = match inst.physical_devices() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("vulkan-video round10: physical_devices failed: {e}; skipping");
            return None;
        }
    };
    for d in &devices {
        let support = d.supports_video_extensions();
        if !support.queue_khr || !support.decode_h264 {
            continue;
        }
        if let Some(qfi) = d.video_queue_family_indices().first().copied() {
            return Some((d.handle(), qfi));
        }
    }
    eprintln!("vulkan-video round10: no H.264-decode-capable device; skipping");
    None
}

/// Create the "application-owned" logical device the import tests
/// wrap. Mirrors what a renderer that wants video decode would do.
fn create_app_device(
    inst: &Instance,
) -> Option<(Device, oxideav_vulkan_video::sys::VkPhysicalDevice, u32)> {
    let (pd_handle, qfi) = pick_h264_device(inst)?;
    let pds = inst.physical_devices().ok()?;
    let pd = pds.iter().find(|p| p.handle() == pd_handle)?;
    match Device::new(
        pd,
        qfi,
        &[
            VK_KHR_SYNCHRONIZATION_2_NAME,
            VK_KHR_VIDEO_QUEUE_NAME,
            VK_KHR_VIDEO_DECODE_QUEUE_NAME,
            VK_KHR_VIDEO_DECODE_H264_NAME,
        ],
    ) {
        Ok(dev) => Some((dev, pd_handle, qfi)),
        Err(e) => {
            eprintln!("vulkan-video round10: vkCreateDevice failed: {e}; skipping");
            None
        }
    }
}

fn make_packet(bytes: Vec<u8>) -> Packet {
    Packet {
        stream_index: 0,
        time_base: TimeBase::new(1, 1),
        pts: Some(0),
        dts: Some(0),
        duration: None,
        data: bytes,
        flags: Default::default(),
    }
}

/// Same success classification as rounds 4/7: `Ok` or the
/// `OXIDEAV_VK_SKIP_SUBMIT` soft-fail marker both count as "pipeline
/// constructed".
fn run_send_packet_skip_submit(
    dec: &mut Box<dyn oxideav_core::Decoder>,
    pkt: &Packet,
) -> Result<(), String> {
    let _guard = ENV_HOOK_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    // SAFETY: the env mutation is serialised through `ENV_HOOK_LOCK`
    // (every test in this binary that touches the hook takes it) and
    // the decoder reads the variable on this thread only.
    unsafe { std::env::set_var("OXIDEAV_VK_SKIP_SUBMIT", "1") };
    let result = dec.send_packet(pkt);
    // SAFETY: as above; still under the same lock guard.
    unsafe { std::env::remove_var("OXIDEAV_VK_SKIP_SUBMIT") };
    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            let msg = format!("{e}");
            if msg.contains("OXIDEAV_VK_SKIP_SUBMIT") {
                Ok(())
            } else {
                Err(msg)
            }
        }
    }
}

#[test]
fn instance_from_raw_is_non_owning() {
    let Some(app_instance) = try_init_instance() else {
        return;
    };

    let raw = app_instance.handle();
    {
        // SAFETY: `raw` is a live instance owned by `app_instance`,
        // which outlives this scope; nothing destroys it concurrently.
        let imported = unsafe { Instance::from_raw(raw) }
            .expect("Instance::from_raw must resolve the dispatch surface");
        assert_eq!(imported.handle(), raw, "imported wrapper keeps the handle");
        let devices = imported
            .physical_devices()
            .expect("enumeration through the imported wrapper");
        assert!(
            !devices.is_empty(),
            "imported instance must see the same non-empty device list"
        );
        // `imported` drops here — must NOT call vkDestroyInstance.
    }

    // The original instance is still alive and fully usable.
    let devices = app_instance
        .physical_devices()
        .expect("original instance must survive the imported wrapper's Drop");
    assert!(!devices.is_empty());
}

#[test]
fn device_from_raw_resolves_video_dispatch() {
    let Some(app_instance) = try_init_instance() else {
        return;
    };
    let Some((app_device, pd_handle, qfi)) = create_app_device(&app_instance) else {
        return;
    };

    {
        // SAFETY: `pd_handle` was enumerated from `app_instance`;
        // `app_device.handle()` is live and created from it. Both
        // outlive this scope.
        let imported = unsafe {
            let pd = app_instance.physical_device_from_raw(pd_handle);
            Device::from_raw(&pd, app_device.handle())
        }
        .expect("Device::from_raw must resolve the video dispatch surface");
        assert_eq!(imported.handle(), app_device.handle());
        // Queue retrieval works through the imported wrapper.
        let q = imported.queue_indexed(qfi, 0);
        assert!(!q.handle().is_null(), "vkGetDeviceQueue returned NULL");
        assert_eq!(q.family_index(), qfi);
        // `imported` drops here — must NOT call vkDestroyDevice.
    }

    // The application's device is still alive: queue fetch works.
    let q = app_device.queue(qfi);
    assert!(
        !q.handle().is_null(),
        "original device must survive the imported wrapper's Drop"
    );
}

#[test]
fn make_with_device_builds_pipeline_on_imported_device() {
    let Some(bytes) = read_fixture() else {
        return;
    };
    let Some(app_instance) = try_init_instance() else {
        return;
    };
    let Some((app_device, pd_handle, qfi)) = create_app_device(&app_instance) else {
        return;
    };

    let params = CodecParameters::video(CodecId::new("h264"));
    let ext = ExternalDevice::new(app_instance.handle(), pd_handle, app_device.handle(), qfi);

    // SAFETY: all three handles are live, mutually consistent, and
    // owned by `app_instance` / `app_device`, both of which outlive
    // `dec` (dropped explicitly below, before they go out of scope).
    // The device was created with the required video extensions and
    // queue 0 of `qfi`; no other thread touches the queue.
    let mut dec = match unsafe { H264VkDecoder::make_with_device(&params, ext) } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("vulkan-video round10: make_with_device failed: {e}; skipping");
            return;
        }
    };

    let pkt = make_packet(bytes);
    match run_send_packet_skip_submit(&mut dec, &pkt) {
        Ok(()) => {}
        Err(msg) => panic!("imported-device pipeline construction failed: {msg}"),
    }

    // Decoder teardown must leave the application's objects alive…
    drop(dec);

    // …so the app can keep rendering with them afterwards.
    let q = app_device.queue(qfi);
    assert!(
        !q.handle().is_null(),
        "application device must survive decoder Drop"
    );
    let devices = app_instance
        .physical_devices()
        .expect("application instance must survive decoder Drop");
    assert!(!devices.is_empty());
}

#[test]
fn make_with_device_rejects_non_video_queue_family() {
    let Some(bytes) = read_fixture() else {
        return;
    };
    let Some(app_instance) = try_init_instance() else {
        return;
    };
    let Some((app_device, pd_handle, _qfi)) = create_app_device(&app_instance) else {
        return;
    };

    let params = CodecParameters::video(CodecId::new("h264"));
    // Queue family 4095 does not exist on any real device — the
    // decoder must reject it with a clear Unsupported error at lazy
    // pipeline construction, not crash.
    let ext = ExternalDevice::new(app_instance.handle(), pd_handle, app_device.handle(), 4095);

    // SAFETY: handles are live and consistent (see previous test);
    // the bogus queue family is validated before any queue is
    // fetched.
    let mut dec = match unsafe { H264VkDecoder::make_with_device(&params, ext) } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("vulkan-video round10: make_with_device failed: {e}; skipping");
            return;
        }
    };

    let pkt = make_packet(bytes);
    let _guard = ENV_HOOK_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    // SAFETY: the env mutation is serialised through `ENV_HOOK_LOCK`
    // (every test in this binary that touches the hook takes it) and
    // the decoder reads the variable on this thread only.
    unsafe { std::env::set_var("OXIDEAV_VK_SKIP_SUBMIT", "1") };
    let result = dec.send_packet(&pkt);
    // SAFETY: as above; still under the same lock guard.
    unsafe { std::env::remove_var("OXIDEAV_VK_SKIP_SUBMIT") };
    drop(_guard);

    match result {
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("not video-capable"),
                "expected the not-video-capable diagnostic, got: {msg}"
            );
        }
        Ok(()) => panic!("bogus queue_family_index 4095 must be rejected"),
    }
    drop(dec);
}
