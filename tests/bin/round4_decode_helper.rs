//! Round 4 decode-helper subprocess.
//!
//! Runs the full Vulkan Video H.264 decode pipeline. Because the
//! NVIDIA proprietary driver currently SIGSEGVs inside
//! `libnvidia-glcore` when `vkQueueSubmit` executes a
//! `vkCmdDecodeVideoKHR` for our test fixture, we run this helper in
//! a separate process so the parent test can absorb the crash and
//! report the milestone reached.
//!
//! Behaviour:
//!   * Reads `OXIDEAV_VK_FIXTURE` (path to an Annex-B .h264 file).
//!   * Decodes the first packet as a single-frame IDR.
//!   * On success, writes a planar I420 dump to
//!     `OXIDEAV_VK_DECODE_OUTPUT` and exits 0.
//!   * On any non-driver failure exits with code 2 and prints the
//!     error to stderr.
//!   * If the driver crashes, the process is killed by the signal and
//!     the parent's `Command::status()` reports `signal: 11`.

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn main() {
    eprintln!("round4_decode_helper: Vulkan Video is Linux/Windows-only; nothing to do here.");
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::path::PathBuf;
#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::process::ExitCode;

#[cfg(any(target_os = "linux", target_os = "windows"))]
use oxideav_core::{time::TimeBase, CodecId, CodecParameters, Frame, Packet};
#[cfg(any(target_os = "linux", target_os = "windows"))]
use oxideav_vulkan_video::decoder::H264VkDecoder;

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn main() -> ExitCode {
    let fixture = match std::env::var("OXIDEAV_VK_FIXTURE") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("helper: OXIDEAV_VK_FIXTURE not set");
            return ExitCode::from(2);
        }
    };
    let output = match std::env::var("OXIDEAV_VK_DECODE_OUTPUT") {
        Ok(p) => PathBuf::from(p),
        Err(_) => {
            eprintln!("helper: OXIDEAV_VK_DECODE_OUTPUT not set");
            return ExitCode::from(2);
        }
    };

    let bytes = match std::fs::read(&fixture) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("helper: read {fixture:?}: {e}");
            return ExitCode::from(2);
        }
    };

    let params = CodecParameters::video(CodecId::new("h264"));
    let mut dec = match H264VkDecoder::make(&params) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("helper: decoder make failed: {e}");
            return ExitCode::from(2);
        }
    };

    let pkt = Packet {
        stream_index: 0,
        time_base: TimeBase::new(1, 1),
        pts: Some(0),
        dts: Some(0),
        duration: None,
        data: bytes,
        flags: Default::default(),
    };

    if let Err(e) = dec.send_packet(&pkt) {
        eprintln!("helper: send_packet returned {e}");
        return ExitCode::from(2);
    }

    let frame = match dec.receive_frame() {
        Ok(Frame::Video(f)) => f,
        Ok(_) => {
            eprintln!("helper: received non-video frame");
            return ExitCode::from(2);
        }
        Err(e) => {
            eprintln!("helper: receive_frame returned {e}");
            return ExitCode::from(2);
        }
    };

    // Serialise as I420 (Y + U + V).
    let mut buf = Vec::new();
    for plane in &frame.planes {
        buf.extend_from_slice(&plane.data);
    }
    if let Err(e) = std::fs::write(&output, &buf) {
        eprintln!("helper: write {output:?}: {e}");
        return ExitCode::from(2);
    }

    eprintln!("helper: wrote {} bytes to {output:?}", buf.len());
    ExitCode::SUCCESS
}
