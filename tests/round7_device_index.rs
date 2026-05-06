//! Round 7 integration tests — `CodecParameters::device_index` plumbing.
//!
//! Same skip-on-no-Vulkan policy as the earlier rounds: every test
//! reads the SPS+PPS+IDR fixture under `tests/fixtures/`, opens the
//! Vulkan loader, and exits cleanly (without failing) when the host
//! has no Vulkan ICD installed or the fixture is missing — we don't
//! want a clean dev box without GPU access to make CI red.
//!
//! What's covered:
//!
//! * `device_index = None` resolves to "device 0" — the first
//!   physical device that survives `engine_info()`'s filter.
//! * `device_index = Some(0)` is accepted explicitly.
//! * `device_index` past the end of the filtered device list is
//!   reported via `Error::Unsupported` so the framework registry can
//!   fall back to the pure-Rust path.
//!
//! A "device_index = 1" test would only be meaningful on a multi-GPU
//! host and isn't testable on the dev box (single RTX 5080). The
//! out-of-range test (`device_index = 99`) is the deterministic
//! cross-platform signal that the index is honoured at all.
//!
//! The known-issue NVIDIA SIGSEGV at `vkQueueSubmit` time is absorbed
//! the same way `round4_decode.rs` absorbs it: every test here uses
//! the `OXIDEAV_VK_SKIP_SUBMIT` env hook so the soft-fail
//! `OXIDEAV_VK_SKIP_SUBMIT set` error message is treated as success
//! (the pipeline was constructed and recorded; that's exactly what
//! we're asserting — that the right physical device was bound to
//! before the submit ever happens).

#![cfg(any(target_os = "linux", target_os = "windows"))]
#![cfg(feature = "registry")]

use std::path::PathBuf;
use std::sync::Mutex;

use oxideav_core::{time::TimeBase, CodecId, CodecParameters, Packet};

use oxideav_vulkan_video::decoder::H264VkDecoder;

/// `set_var` / `remove_var` are process-global; cargo runs integration
/// tests in parallel by default, and two threads racing on
/// `OXIDEAV_VK_SKIP_SUBMIT` can let a real `vkQueueSubmit` slip through
/// — which on NVIDIA drivers triggers the same `libnvidia-glcore`
/// SIGSEGV that the round 4 helper isolates in a subprocess. This
/// per-test mutex serialises the env-hooked code paths so a fresh
/// `set_var` is guaranteed in scope until the matching `remove_var`.
static ENV_HOOK_LOCK: Mutex<()> = Mutex::new(());

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn read_fixture() -> Option<Vec<u8>> {
    let p = fixtures_dir().join("h264_high_320x240_1frame.h264");
    match std::fs::read(&p) {
        Ok(b) => Some(b),
        Err(_) => {
            eprintln!(
                "vulkan-video round7: fixture missing at {:?}; skipping",
                p
            );
            None
        }
    }
}

fn vulkan_available() -> bool {
    if oxideav_vulkan_video::sys::framework().is_err() {
        eprintln!("vulkan-video round7: no Vulkan loader; skipping");
        return false;
    }
    true
}

/// Returns `true` if `engine_info()` reports at least one device on
/// this host. Used to gate the success-path tests so a "no ICD
/// installed" CI box runs them as a no-op rather than a hard fail.
fn has_engine_devices() -> bool {
    let devs = oxideav_vulkan_video::engine_info();
    if devs.is_empty() {
        eprintln!("vulkan-video round7: no engine_info devices; skipping");
        return false;
    }
    true
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

/// Drive `dec.send_packet(pkt)` under `OXIDEAV_VK_SKIP_SUBMIT=1` and
/// classify the result the same way `round4_decode::h264_decoder_constructs_full_pipeline`
/// does:
///
/// * `Ok(())` — pipeline constructed & submit suppressed without an
///   error; this is treated as a success.
/// * `Err(e)` containing the `OXIDEAV_VK_SKIP_SUBMIT` marker — the
///   pipeline reached the submit hook and bailed cleanly; success.
/// * any other `Err(e)` — propagated as failure.
fn run_send_packet_skip_submit(
    dec: &mut Box<dyn oxideav_core::Decoder>,
    pkt: &Packet,
) -> Result<(), String> {
    let _guard = ENV_HOOK_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    std::env::set_var("OXIDEAV_VK_SKIP_SUBMIT", "1");
    let result = dec.send_packet(pkt);
    std::env::remove_var("OXIDEAV_VK_SKIP_SUBMIT");
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
fn device_index_none_uses_first_video_device() {
    let Some(bytes) = read_fixture() else { return; };
    if !vulkan_available() {
        return;
    }
    if !has_engine_devices() {
        return;
    }

    // No `with_device_index` — the field stays `None` and the factory
    // resolves it to `0` via `unwrap_or(0)`.
    let params = CodecParameters::video(CodecId::new("h264"));
    assert!(params.device_index.is_none(), "default must be None");

    let mut dec = match H264VkDecoder::make(&params) {
        Ok(d) => d,
        Err(e) => {
            // No Vulkan ICD path — same skip semantics as round 4.
            eprintln!("vulkan-video round7: make failed: {e}; skipping");
            return;
        }
    };

    let pkt = make_packet(bytes);
    match run_send_packet_skip_submit(&mut dec, &pkt) {
        Ok(()) => {}
        Err(e) => panic!(
            "expected the default device_index (None → 0) to construct the pipeline; got: {e}"
        ),
    }
}

#[test]
fn device_index_zero_explicit_works() {
    let Some(bytes) = read_fixture() else { return; };
    if !vulkan_available() {
        return;
    }
    if !has_engine_devices() {
        return;
    }

    let params = CodecParameters::video(CodecId::new("h264")).with_device_index(0);
    assert_eq!(params.device_index, Some(0));

    let mut dec = match H264VkDecoder::make(&params) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("vulkan-video round7: make failed: {e}; skipping");
            return;
        }
    };

    let pkt = make_packet(bytes);
    match run_send_packet_skip_submit(&mut dec, &pkt) {
        Ok(()) => {}
        Err(e) => panic!(
            "expected an explicit with_device_index(0) to construct the pipeline; got: {e}"
        ),
    }
}

#[test]
fn device_index_out_of_range_errors() {
    let Some(bytes) = read_fixture() else { return; };
    if !vulkan_available() {
        return;
    }
    // No has_engine_devices() gate: even on a host where engine_info()
    // reports zero devices, an out-of-range index must surface as an
    // error (0 >= 0 is still out of range).

    // 99 is far beyond any plausible single- or multi-GPU host.
    let params = CodecParameters::video(CodecId::new("h264")).with_device_index(99);
    let mut dec = match H264VkDecoder::make(&params) {
        Ok(d) => d,
        Err(e) => {
            // make() only fails when the loader is missing — we already
            // checked vulkan_available() above, so this is a real
            // problem.
            panic!("vulkan-video round7: make should succeed even for an out-of-range index — index validation happens lazily on first SPS/PPS: {e}");
        }
    };

    // The error fires lazily inside `ensure_state` on the first
    // SPS+PPS pair seen — feed the IDR packet and expect Err.
    let pkt = make_packet(bytes);
    let result = {
        let _guard = ENV_HOOK_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        std::env::set_var("OXIDEAV_VK_SKIP_SUBMIT", "1");
        let r = dec.send_packet(&pkt);
        std::env::remove_var("OXIDEAV_VK_SKIP_SUBMIT");
        r
    };
    let err = result.expect_err(
        "device_index=99 must produce an error from ensure_state / DecoderState::create",
    );
    let msg = format!("{err}");
    assert!(
        msg.contains("device_index 99") && msg.contains("out of range"),
        "expected an out-of-range diagnostic mentioning device_index 99; got: {msg}"
    );
}

#[test]
fn device_index_default_is_none_in_codec_parameters() {
    // Sanity sibling assertion — locked in here so a future
    // refactor of `CodecParameters` doesn't silently break the
    // device_index plumbing this test file exercises.
    let p = CodecParameters::video(CodecId::new("h264"));
    assert!(p.device_index.is_none());
    let p2 = p.with_device_index(7);
    assert_eq!(p2.device_index, Some(7));
}
