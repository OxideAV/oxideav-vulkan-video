//! Vulkan Video H.264 decoder pipeline.
//!
//! Round 4 implementation. The decoder is a packet-driven `oxideav_core::Decoder`
//! impl that lazily constructs the `VkInstance` / `VkDevice` /
//! `VkVideoSessionKHR` / `VkVideoSessionParametersKHR` / images / buffers
//! / command pool the first time SPS+PPS are seen, then issues a
//! single `vkCmdDecodeVideoKHR` per VCL slice and copies the decoded
//! NV12 image back into a planar `VideoFrame`.
//!
//! # Pipeline shape
//!
//! 1. Parse Annex-B SPS / PPS into the `StdVideoH264*` structs the GPU
//!    consumes (delegated to the workspace-shared
//!    [`oxideav_bitstream::h264`] parser).
//! 2. Open a `VkInstance` (Vulkan 1.2), pick a discrete GPU that
//!    advertises `VK_KHR_video_decode_h264`, and create a `VkDevice`
//!    with a queue from a video-decode-capable queue family.
//! 3. Query H.264 decode capabilities to know the std-header version,
//!    DPB-slot upper bound, and bitstream alignment.
//! 4. Build a `VkVideoSessionKHR`, allocate + bind its memory backing.
//! 5. Build a `VkVideoSessionParametersKHR` carrying the parsed
//!    SPS + PPS.
//! 6. Allocate a 2D-array `VkImage` for the DPB (NV12 layout, layer
//!    count = `max_dpb_slots`), one or more output `VkImage`s sharing
//!    the same array layout for output-coincide implementations, and
//!    bind device memory to each.
//! 7. Allocate a host-visible `VkBuffer` for the H.264 bitstream
//!    payload (the full Annex-B picture, including SPS+PPS+slice).
//!    Note: NVIDIA's Vulkan driver accepts Annex-B start codes inside
//!    the bitstream payload — no MP4-AVCC reframing required.
//! 8. Allocate a host-visible staging `VkBuffer` sized for one NV12
//!    frame (luma + chroma plane) so we can read pixels back via
//!    `vkCmdCopyImageToBuffer`.
//! 9. Record + submit a single command buffer that:
//!    a. Transitions the DPB image to `VIDEO_DECODE_DPB_KHR`.
//!    b. Begins coding scope.
//!    c. Issues the spec-mandated `RESET` control on first submission.
//!    d. Issues `vkCmdDecodeVideoKHR` against the SPS/PPS-bound
//!    `VkVideoSessionParametersKHR`, with a setup reference slot
//!    identifying DPB slot 0 (the IDR's reconstruction).
//!    e. Ends coding scope.
//!    f. Transitions the output image to `TRANSFER_SRC_OPTIMAL` and
//!    `vkCmdCopyImageToBuffer` it to staging.
//! 10. Submit, wait via `vkQueueWaitIdle`, then memcpy from the
//!     mapped staging buffer into a planar `VideoFrame`.
//!
//! # Reality check
//!
//! Vulkan video decode is the most fragile of the four bridges. Driver
//! quirks abound (NVIDIA wants the bitstream pre-padded to a specific
//! alignment, AMD wants distinct DPB / output images, Intel returns
//! `OUT_OF_DATE_KHR` from `vkCmdDecodeVideoKHR` in some firmware
//! versions). The pipeline above is the spec-recommended shape; if a
//! driver-specific quirk causes the test to come back with constant
//! pixels instead of the rendered IDR, we record that in CHANGELOG and
//! fall through to a `Error::Unsupported` runtime registration so the
//! framework's pure-Rust h264 path takes over.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr;

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame, Packet, Result, VideoFrame, VideoPlane,
};

use oxideav_bitstream::h264::{
    self as bs_h264, H264Pps, H264Sps, NAL_TYPE_IDR, NAL_TYPE_NON_IDR_SLICE, NAL_TYPE_PPS,
    NAL_TYPE_SPS,
};

use crate::device::Device;
use crate::instance::Instance;
use crate::physical_device::{
    PhysicalDevice, PhysicalDeviceType, VK_KHR_VIDEO_DECODE_H264_NAME,
    VK_KHR_VIDEO_DECODE_QUEUE_NAME, VK_KHR_VIDEO_QUEUE_NAME,
};
use crate::sys::{
    self, StdVideoDecodeH264PictureInfo, StdVideoDecodeH264PictureInfoFlags,
    StdVideoDecodeH264ReferenceInfo, StdVideoH264PictureParameterSet, StdVideoH264PpsFlags,
    StdVideoH264SequenceParameterSet, StdVideoH264SpsFlags, VkBuffer, VkBufferCreateInfo,
    VkBufferImageCopy, VkCommandBuffer, VkCommandBufferAllocateInfo, VkCommandBufferBeginInfo,
    VkCommandPool, VkCommandPoolCreateInfo, VkComponentMapping, VkDeviceMemory, VkExtent2D,
    VkExtent3D, VkImage, VkImageCreateInfo, VkImageMemoryBarrier, VkImageSubresourceLayers,
    VkImageSubresourceRange, VkImageView, VkImageViewCreateInfo, VkMemoryAllocateInfo,
    VkMemoryRequirements, VkOffset2D, VkOffset3D, VkPhysicalDeviceMemoryProperties, VkSubmitInfo,
    VkVideoBeginCodingInfoKHR, VkVideoCodingControlInfoKHR, VkVideoDecodeH264DpbSlotInfoKHR,
    VkVideoDecodeH264PictureInfoKHR, VkVideoDecodeH264ProfileInfoKHR,
    VkVideoDecodeH264SessionParametersAddInfoKHR, VkVideoDecodeH264SessionParametersCreateInfoKHR,
    VkVideoDecodeInfoKHR, VkVideoEndCodingInfoKHR, VkVideoPictureResourceInfoKHR,
    VkVideoProfileInfoKHR, VkVideoProfileListInfoKHR, VkVideoReferenceSlotInfoKHR,
    VkVideoSessionParametersCreateInfoKHR, VkVideoSessionParametersKHR,
    STD_VIDEO_H264_CHROMA_FORMAT_IDC_420, STD_VIDEO_H264_PROFILE_IDC_HIGH,
    VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
    VK_API_VERSION_1_2, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
    VK_IMAGE_ASPECT_PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR,
    VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_TYPE_2D,
    VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR,
    VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
    VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_QUEUE_FAMILY_IGNORED, VK_SAMPLE_COUNT_1_BIT, VK_SHARING_MODE_EXCLUSIVE,
    VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
    VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    VK_STRUCTURE_TYPE_SUBMIT_INFO, VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR, VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR, VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR, VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR, VK_SUCCESS,
    VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR, VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
    VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR, VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR,
};
use crate::video::{query_video_decode_h264_capabilities, VideoSession};

// ─────────────────────────── helpers ─────────────────────────────────────────

fn vk_err(op: &'static str, r: i32) -> Error {
    Error::other(format!("{op} returned VkResult({r})"))
}

/// Round `v` up to the nearest multiple of `align`. Caller ensures
/// `align != 0`.
fn align_up(v: u64, align: u64) -> u64 {
    if align == 0 {
        v
    } else {
        v.div_ceil(align) * align
    }
}

/// Walk the Annex-B bitstream and return the byte offsets (relative
/// to the start of `bitstream`) of each VCL slice's start-code prefix.
///
/// VCL slices are NAL types 1 (non-IDR slice) and 5 (IDR slice). The
/// emitted offset points to the FIRST byte of the start-code prefix
/// (0x000001 or 0x00000001).
fn compute_slice_offsets(bitstream: &[u8]) -> Vec<u32> {
    let mut offsets = Vec::new();
    let mut pos = 0usize;
    let len = bitstream.len();

    while pos + 4 <= len {
        let sc_pos;
        let sc_len;
        if bitstream[pos] == 0
            && bitstream[pos + 1] == 0
            && bitstream[pos + 2] == 0
            && bitstream[pos + 3] == 1
        {
            sc_pos = pos;
            sc_len = 4;
        } else if bitstream[pos] == 0 && bitstream[pos + 1] == 0 && bitstream[pos + 2] == 1 {
            sc_pos = pos;
            sc_len = 3;
        } else {
            pos += 1;
            continue;
        }
        let nal_byte_pos = sc_pos + sc_len;
        if nal_byte_pos < len {
            let nt = bitstream[nal_byte_pos] & 0x1F;
            if nt == 1 || nt == 5 {
                offsets.push(sc_pos as u32);
            }
        }
        pos = sc_pos + sc_len;
    }
    offsets
}

/// Whether `pd` would be surfaced by [`crate::engine_info`]. The
/// decoder uses the same predicate so `device_index` indexes into the
/// same filtered list a CLI consumer of `engine_info()` sees.
///
/// Mirrors `crate::engine::build_device_info`'s admit rule:
/// `device_type ∈ {Discrete, Integrated, Virtual} GPU OR any
/// VK_KHR_video_* extension advertised`. CPU / Other ICDs without
/// any video extension are skipped — the same set engine_info()
/// would have skipped.
fn engine_info_filter_admits(pd: &PhysicalDevice<'_>) -> bool {
    let props = pd.properties();
    let useful_type = matches!(
        props.device_type,
        PhysicalDeviceType::DiscreteGpu
            | PhysicalDeviceType::IntegratedGpu
            | PhysicalDeviceType::VirtualGpu
    );
    if useful_type {
        return true;
    }
    let v = pd.supports_video_extensions();
    v.queue_khr || v.decode_h264 || v.decode_h265 || v.decode_av1 || v.encode_h264 || v.encode_h265
}

/// Pick the first memory type whose bit is set in `type_bits` AND
/// whose flags satisfy `required`.
fn pick_memory_type(
    props: &VkPhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: u32,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if type_bits & (1u32 << i) == 0 {
            continue;
        }
        if (props.memory_types[i as usize].property_flags & required) == required {
            return Some(i);
        }
    }
    None
}

/// Wrapper that owns the Vulkan objects backing one decode session.
///
/// Drop tears them down in the order the spec requires: command pool +
/// fence first, then images / image-views / buffers, then session
/// parameters, then session, then memory, then device, then instance.
/// Per-instance capability data captured at construction time so we
/// don't need to re-query during every decode_picture call.
#[allow(dead_code)]
struct CachedCaps {
    bitstream_offset_alignment: u64,
    bitstream_size_alignment: u64,
}

struct DecoderState {
    // ─── Drop order matters! ──────────────────────────────────────
    // Field declaration order = drop order. The Vulkan spec requires
    // children to be destroyed before their parents:
    //   command_buffer → command_pool → buffers/images/views/memory
    //   → session_params → session → device → instance.
    // Putting children FIRST and parents LAST gives us that order.
    // We do explicit destruction in `Drop` (because most of the
    // Vulkan handles are non-RAII raw types) but we still rely on
    // `device` and `instance` being kept alive until the very end.
    /// Pre-recorded command buffer for one decode dispatch. Freed
    /// before `command_pool` in Drop.
    command_buffer: VkCommandBuffer,
    command_pool: VkCommandPool,

    bitstream_buffer: VkBuffer,
    bitstream_memory: VkDeviceMemory,
    bitstream_size: u64,

    staging_buffer: VkBuffer,
    staging_memory: VkDeviceMemory,
    staging_size: u64,

    output_image_view: VkImageView,
    output_image: VkImage,
    output_memory: VkDeviceMemory,

    dpb_image_view: VkImageView,
    dpb_image: VkImage,
    dpb_memory: VkDeviceMemory,

    session_params: VkVideoSessionParametersKHR,
    /// Owns `vkDestroyVideoSessionKHR`. Must drop before `device`.
    session: Option<VideoSession<'static>>,

    queue_family_index: u32,
    /// Held for diagnostics / future round expansion.
    #[allow(dead_code)]
    physical_device_handle: sys::VkPhysicalDevice,

    /// Owns `vkDestroyDevice`. Must drop before `instance`.
    device: Device,

    /// Owns `vkDestroyInstance`. Drops last.
    #[allow(dead_code)]
    instance: Instance,

    width: u32,
    height: u32,
    /// Stride in bytes used for staging (luma row pitch, also chroma
    /// row pitch since both are byte-stepped at width for NV12).
    luma_stride: u32,
    chroma_stride: u32,
    chroma_height: u32,

    /// Whether DPB and output coincide on this driver. NVIDIA on
    /// Linux reports `DPB_AND_OUTPUT_COINCIDE_BIT_KHR` and we use the
    /// same array image for both. AMD historically reports `DISTINCT`,
    /// in which case `output_image` is its own image.
    coincide: bool,

    /// Bitstream buffer offset alignment from caps. Currently unused
    /// (we use the size alignment for the decode srcBufferRange) but
    /// kept for future round expansion (multi-slice / multi-picture
    /// streams will need explicit offset alignment).
    #[allow(dead_code)]
    bitstream_offset_alignment: u64,
    /// Bitstream buffer size alignment from caps.
    bitstream_size_alignment: u64,
}

// SAFETY: every Vulkan handle in `DecoderState` is externally
// synchronised — we only ever drive them from `&mut self` so there
// are no aliased mutations from other threads. The Device / Instance
// already encapsulate the same single-threaded contract.
unsafe impl Send for DecoderState {}

impl Drop for DecoderState {
    fn drop(&mut self) {
        // SAFETY: we hold an exclusive `&mut self` for the duration of
        // Drop; the device handle is still alive (Device's Drop runs
        // after this on field declaration order).
        let dfns: &crate::device::DeviceFns = self.device.fns();

        // Wait for any pending work.
        unsafe {
            (dfns.queue_wait_idle)(self.device.queue(self.queue_family_index).handle());
        }

        unsafe {
            if !self.command_buffer.is_null() && !self.command_pool.is_null() {
                (dfns.free_command_buffers)(
                    self.device.handle(),
                    self.command_pool,
                    1,
                    &self.command_buffer,
                );
            }
            if !self.command_pool.is_null() {
                (dfns.destroy_command_pool)(self.device.handle(), self.command_pool, ptr::null());
            }
            if !self.bitstream_buffer.is_null() {
                (dfns.destroy_buffer)(self.device.handle(), self.bitstream_buffer, ptr::null());
            }
            if !self.bitstream_memory.is_null() {
                (dfns.free_memory)(self.device.handle(), self.bitstream_memory, ptr::null());
            }
            if !self.staging_buffer.is_null() {
                (dfns.destroy_buffer)(self.device.handle(), self.staging_buffer, ptr::null());
            }
            if !self.staging_memory.is_null() {
                (dfns.free_memory)(self.device.handle(), self.staging_memory, ptr::null());
            }
            if !self.output_image_view.is_null() {
                (dfns.destroy_image_view)(
                    self.device.handle(),
                    self.output_image_view,
                    ptr::null(),
                );
            }
            if !self.output_image.is_null() && self.output_image != self.dpb_image {
                (dfns.destroy_image)(self.device.handle(), self.output_image, ptr::null());
            }
            if !self.output_memory.is_null() && self.output_memory != self.dpb_memory {
                (dfns.free_memory)(self.device.handle(), self.output_memory, ptr::null());
            }
            if !self.dpb_image_view.is_null() {
                (dfns.destroy_image_view)(self.device.handle(), self.dpb_image_view, ptr::null());
            }
            if !self.dpb_image.is_null() {
                (dfns.destroy_image)(self.device.handle(), self.dpb_image, ptr::null());
            }
            if !self.dpb_memory.is_null() {
                (dfns.free_memory)(self.device.handle(), self.dpb_memory, ptr::null());
            }
            if !self.session_params.is_null() {
                (dfns.destroy_video_session_parameters_khr)(
                    self.device.handle(),
                    self.session_params,
                    ptr::null(),
                );
            }
        }

        // VideoSession's Drop fires when self.session is dropped; that
        // happens before self.device because field declaration order
        // is preserved on drop.
        self.session = None;
    }
}

// ─────────────────────────── public decoder ──────────────────────────────────

/// Vulkan Video H.264 decoder.
///
/// Implements [`oxideav_core::Decoder`] by deferring heavy initialisation
/// (instance / device / session / parameters / images / buffers) until
/// the first SPS+PPS NAL pair is seen on `send_packet`.
pub struct H264VkDecoder {
    codec_id: CodecId,
    state: Option<DecoderState>,
    sps_nals: Vec<Vec<u8>>,
    pps_nals: Vec<Vec<u8>>,
    output_queue: VecDeque<VideoFrame>,
    flushed: bool,
    /// Captured at construction from
    /// [`CodecParameters::device_index`] (`unwrap_or(0)`). Indexes
    /// into the same filtered physical-device list that
    /// [`crate::engine_info`] reports.
    device_index: u32,
}

impl H264VkDecoder {
    /// Factory used by `register()`.
    ///
    /// Honours [`CodecParameters::device_index`] (`unwrap_or(0)`) by
    /// capturing the requested index here; the index is consumed when
    /// the heavy [`DecoderState`] is lazily constructed on the first
    /// SPS+PPS NAL pair. `device_index` is interpreted against the
    /// **same filter** [`crate::engine_info`] applies — every physical
    /// device whose type is `Discrete` / `Integrated` / `Virtual` GPU
    /// or that advertises any `VK_KHR_video_*` extension. An out-of-
    /// range index is reported via [`Error::Unsupported`] so the codec
    /// registry can fall back to a software path.
    pub fn make(params: &CodecParameters) -> Result<Box<dyn oxideav_core::Decoder>> {
        // Probe the loader so we fail fast on hosts without Vulkan
        // (the framework registry will fall back to the pure-Rust
        // path).
        sys::vtable().map_err(|e| Error::unsupported(format!("vulkan-video: {e}")))?;
        let device_index = params.device_index.unwrap_or(0);
        Ok(Box::new(H264VkDecoder {
            codec_id: CodecId::new("h264"),
            state: None,
            sps_nals: Vec::new(),
            pps_nals: Vec::new(),
            output_queue: VecDeque::new(),
            flushed: false,
            device_index,
        }))
    }

    fn ensure_state(&mut self, sps: &H264Sps, pps: &H264Pps) -> Result<()> {
        if self.state.is_some() {
            return Ok(());
        }
        let st = DecoderState::create(sps, pps, self.device_index)?;
        self.state = Some(st);
        Ok(())
    }

    /// Submit one Annex-B picture (SPS+PPS+slice) to the GPU and copy
    /// the decoded NV12 frame back into `self.output_queue`.
    fn submit_picture(&mut self, picture_bytes: &[u8], sps: &H264Sps) -> Result<()> {
        let state = self.state.as_mut().ok_or_else(|| {
            Error::other("vulkan-video: ensure_state must be called before submit_picture")
        })?;
        state.decode_picture(picture_bytes, sps, &mut self.output_queue)
    }
}

unsafe impl Send for H264VkDecoder {}

impl oxideav_core::Decoder for H264VkDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        self.flushed = false;

        let nals = bs_h264::split_annex_b(&packet.data);

        // Collect the full Annex-B bitstream (including SPS/PPS) so the
        // GPU can re-parse from raw H.264 bytes; we still extract the
        // structured SPS / PPS for `VkVideoSessionParametersKHR` lookup.
        let mut got_params = false;
        for nal in &nals {
            if nal.is_empty() {
                continue;
            }
            let nt = nal[0] & 0x1F;
            match nt {
                NAL_TYPE_SPS => {
                    self.sps_nals.clear();
                    self.sps_nals.push(nal.to_vec());
                    got_params = true;
                }
                NAL_TYPE_PPS => {
                    self.pps_nals.clear();
                    self.pps_nals.push(nal.to_vec());
                    got_params = true;
                }
                _ => {}
            }
        }

        if got_params {
            let sps_nal = self.sps_nals.first().cloned();
            let pps_nal = self.pps_nals.first().cloned();
            if let (Some(s), Some(p)) = (sps_nal, pps_nal) {
                let parsed_sps = bs_h264::parse_sps_nal(&s)
                    .map_err(|e| Error::other(format!("vulkan-video: SPS parse failed: {e}")))?;
                let parsed_pps = bs_h264::parse_pps_nal(&p)
                    .map_err(|e| Error::other(format!("vulkan-video: PPS parse failed: {e}")))?;
                self.ensure_state(&parsed_sps, &parsed_pps)?;
            }
        }

        // Find the VCL slice NAL — for the IDR fixture we expect one.
        let mut have_vcl = false;
        for nal in &nals {
            if nal.is_empty() {
                continue;
            }
            let nt = nal[0] & 0x1F;
            if nt == NAL_TYPE_IDR || nt == NAL_TYPE_NON_IDR_SLICE {
                have_vcl = true;
                break;
            }
        }

        if !have_vcl {
            return Ok(());
        }

        let parsed_sps = bs_h264::parse_sps_nal(
            self.sps_nals
                .first()
                .ok_or_else(|| Error::other("vulkan-video: VCL slice arrived before SPS"))?,
        )
        .map_err(|e| Error::other(format!("vulkan-video: SPS parse failed: {e}")))?;

        // The "picture bytes" we send to the GPU are the full packet's
        // Annex-B payload (SPS + PPS + slice). NVIDIA's driver
        // tolerates having SPS/PPS in-band; the slice NAL is what
        // drives the decode.
        self.submit_picture(&packet.data, &parsed_sps)?;
        Ok(())
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.output_queue.pop_front() {
            return Ok(Frame::Video(f));
        }
        Err(if self.flushed {
            Error::Eof
        } else {
            Error::NeedMore
        })
    }

    fn flush(&mut self) -> Result<()> {
        self.flushed = true;
        Ok(())
    }
}

// ─────────────────────────── DecoderState — heavy lifting ────────────────────

impl DecoderState {
    fn create(sps: &H264Sps, pps: &H264Pps, device_index: u32) -> Result<Self> {
        // ── Instance ────────────────────────────────────────────
        let instance = Instance::new("oxideav-vulkan-video", VK_API_VERSION_1_2)
            .map_err(|e| Error::unsupported(format!("vulkan-video: {e}")))?;

        // ── Pick a video-decode-capable physical device ─────────
        //
        // `device_index` is interpreted against the *same* filter
        // `engine_info()` applies (see `crate::engine`): every
        // physical device whose type is Discrete/Integrated/Virtual
        // GPU OR that advertises at least one `VK_KHR_video_*`
        // extension is included, in `Instance::physical_devices()`
        // enumeration order. The two MUST stay in sync — an
        // `engine_info()` consumer that prints "device 1" and then
        // passes `with_device_index(1)` expects the decoder to bind
        // to that exact same physical device.
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: enumerating physical devices");
        }
        let devices = instance
            .physical_devices()
            .map_err(|e| Error::unsupported(format!("vulkan-video: {e}")))?;
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: {} physical devices found", devices.len());
        }

        // Filter to the same set engine_info() exposes.
        let filtered_indices: Vec<usize> = devices
            .iter()
            .enumerate()
            .filter(|(_, d)| engine_info_filter_admits(d))
            .map(|(i, _)| i)
            .collect();
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: {} physical device(s) survive engine_info filter",
                filtered_indices.len(),
            );
        }
        if (device_index as usize) >= filtered_indices.len() {
            return Err(Error::unsupported(format!(
                "vulkan-video: device_index {device_index} out of range (0..{})",
                filtered_indices.len()
            )));
        }
        let raw_idx = filtered_indices[device_index as usize];
        let chosen_dev = &devices[raw_idx];
        let support = chosen_dev.supports_video_extensions();
        if !support.queue_khr || !support.decode_h264 {
            return Err(Error::unsupported(format!(
                "vulkan-video: device_index {device_index} does not support H.264 decode \
                 (queue_khr={} decode_h264={})",
                support.queue_khr, support.decode_h264
            )));
        }
        let qfi = chosen_dev
            .video_queue_family_indices()
            .first()
            .copied()
            .ok_or_else(|| {
                Error::unsupported(format!(
                    "vulkan-video: device_index {device_index} advertises decode_h264 \
                     but reports no video-capable queue family"
                ))
            })?;
        let pd_handle = chosen_dev.handle();
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: chosen filtered device_index={} (raw {}) with qfi={}",
                device_index, raw_idx, qfi,
            );
        }
        // Re-enumerate so we don't keep the borrow on `instance` while
        // using `instance` later.
        drop(devices);

        // ── Capabilities ────────────────────────────────────────
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: querying caps");
        }
        let pds = instance
            .physical_devices()
            .map_err(|e| Error::unsupported(format!("vulkan-video: physical_devices: {e}")))?;
        let pd = pds
            .iter()
            .find(|p| p.handle() == pd_handle)
            .ok_or_else(|| Error::other("vulkan-video: physical device disappeared"))?;
        let caps = query_video_decode_h264_capabilities(pd, STD_VIDEO_H264_PROFILE_IDC_HIGH)
            .map_err(|e| Error::unsupported(format!("vulkan-video: caps: {e}")))?;
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: caps OK ({}x{} dpb={} bsalign={})",
                caps.max_coded_extent.0,
                caps.max_coded_extent.1,
                caps.max_dpb_slots,
                caps.min_bitstream_buffer_offset_alignment
            );
        }

        // Driver capability flag: DPB & output coincide?
        let coincide =
            caps.decode_capability_flags & 0x1 != 0 // VK_VIDEO_DECODE_CAPABILITY_DPB_AND_OUTPUT_COINCIDE_BIT_KHR
                ;

        // Memory properties for picking memory types.
        let mem_props = {
            let mut p: VkPhysicalDeviceMemoryProperties = unsafe { std::mem::zeroed() };
            unsafe {
                (pd.instance_fns().get_physical_device_memory_properties)(pd.handle(), &mut p);
            }
            p
        };
        drop(pds);

        // ── Device ──────────────────────────────────────────────
        // We re-enumerate one more time and create the Device.
        let pds = instance
            .physical_devices()
            .map_err(|e| Error::unsupported(format!("vulkan-video: physical_devices2: {e}")))?;
        let pd = pds
            .iter()
            .find(|p| p.handle() == pd_handle)
            .ok_or_else(|| Error::other("vulkan-video: pd lookup2 failed"))?;
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: creating device");
        }
        let device = Device::new(
            pd,
            qfi,
            &[
                VK_KHR_VIDEO_QUEUE_NAME,
                VK_KHR_VIDEO_DECODE_QUEUE_NAME,
                VK_KHR_VIDEO_DECODE_H264_NAME,
            ],
        )
        .map_err(|e| Error::unsupported(format!("vulkan-video: vkCreateDevice: {e}")))?;
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: device OK");
        }
        drop(pds);

        // ── Video session ───────────────────────────────────────
        // We need the lifetime of session and the underlying device
        // to align; we hide this with `transmute` below into an owned
        // `VideoSession<'static>` whose Drop runs before the Device's
        // Drop because of struct field ordering.
        let pds = instance
            .physical_devices()
            .map_err(|e| Error::unsupported(format!("vulkan-video: physical_devices3: {e}")))?;
        let pd = pds
            .iter()
            .find(|p| p.handle() == pd_handle)
            .ok_or_else(|| Error::other("vulkan-video: pd lookup3 failed"))?;
        // We size the session to a generous max so all reasonable test
        // streams fit. Round up to picture-access-granularity.
        let max_w = sps.coded_width().max(caps.min_coded_extent.0);
        let max_h = sps.coded_height().max(caps.min_coded_extent.1);
        let max_w = max_w.min(caps.max_coded_extent.0).max(16);
        let max_h = max_h.min(caps.max_coded_extent.1).max(16);

        let dpb_slots = caps.max_dpb_slots.clamp(1, 17);
        let active_refs = caps.max_active_reference_pictures.min(16);

        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: creating session ({}x{})", max_w, max_h);
        }
        let mut video_session = VideoSession::new_h264_decode(
            // SAFETY: extending `&device` lifetime to 'static. The
            // `Device` is owned by `Self` and freed only after the
            // session in Drop ordering — same struct.
            unsafe { &*(&device as *const Device) },
            pd,
            qfi,
            &caps,
            (max_w, max_h),
            STD_VIDEO_H264_PROFILE_IDC_HIGH,
            dpb_slots,
            active_refs,
        )
        .map_err(|e| Error::unsupported(format!("vulkan-video: vkCreateVideoSessionKHR: {e}")))?;
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: session created, binding memory");
        }
        video_session
            .allocate_and_bind_memory(pd)
            .map_err(|e| Error::other(format!("vulkan-video: bind session memory: {e}")))?;
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: session memory bound");
        }

        drop(pds);

        // ── VkVideoSessionParametersKHR (SPS + PPS) ─────────────
        let std_sps = std_sps_from_parsed(sps);
        let std_pps = std_pps_from_parsed(pps);

        let h264_params_add = VkVideoDecodeH264SessionParametersAddInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
            p_next: ptr::null(),
            std_sps_count: 1,
            p_std_sp_ss: &std_sps,
            std_pps_count: 1,
            p_std_pp_ss: &std_pps,
        };
        let h264_params_create = VkVideoDecodeH264SessionParametersCreateInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
            p_next: ptr::null(),
            max_std_sps_count: 1,
            max_std_pps_count: 1,
            p_parameters_add_info: &h264_params_add,
        };
        let params_create = VkVideoSessionParametersCreateInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
            p_next: &h264_params_create as *const _ as *const c_void,
            flags: 0,
            video_session_parameters_template: ptr::null_mut(),
            video_session: video_session.handle(),
        };
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: creating session parameters");
        }
        let mut session_params: VkVideoSessionParametersKHR = ptr::null_mut();
        let r = unsafe {
            (device.fns().create_video_session_parameters_khr)(
                device.handle(),
                &params_create,
                ptr::null(),
                &mut session_params,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkCreateVideoSessionParametersKHR", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: session parameters OK");
        }

        // ── Profile struct kept for image creation pNext chain ──
        let h264_profile = VkVideoDecodeH264ProfileInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,
            p_next: ptr::null(),
            std_profile_idc: STD_VIDEO_H264_PROFILE_IDC_HIGH,
            picture_layout: VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR,
        };
        let profile = VkVideoProfileInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
            p_next: &h264_profile as *const _ as *const c_void,
            video_codec_operation: VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
            chroma_subsampling: VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
            luma_bit_depth: VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
            chroma_bit_depth: VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
        };
        let profile_list = VkVideoProfileListInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR,
            p_next: ptr::null(),
            profile_count: 1,
            p_profiles: &profile,
        };

        // ── DPB image (2D array, layer per DPB slot) ────────────
        let dpb_layers = dpb_slots;
        let dpb_usage = if coincide {
            VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR | VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR
        } else {
            VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR
        };
        let dpb_image_ci = VkImageCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            p_next: &profile_list as *const _ as *const c_void,
            flags: 0,
            image_type: VK_IMAGE_TYPE_2D,
            format: VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
            extent: VkExtent3D {
                width: max_w,
                height: max_h,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: dpb_layers,
            samples: VK_SAMPLE_COUNT_1_BIT,
            tiling: VK_IMAGE_TILING_OPTIMAL,
            usage: dpb_usage,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: VK_IMAGE_LAYOUT_UNDEFINED,
        };
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: creating DPB image");
        }
        let mut dpb_image: VkImage = ptr::null_mut();
        let r = unsafe {
            (device.fns().create_image)(device.handle(), &dpb_image_ci, ptr::null(), &mut dpb_image)
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkCreateImage(DPB)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: DPB image OK");
        }

        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: DPB image mem requirements");
        }
        let dpb_mem_reqs = {
            let mut req = VkMemoryRequirements::default();
            unsafe {
                (device.fns().get_image_memory_requirements)(device.handle(), dpb_image, &mut req)
            };
            req
        };
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: DPB mem reqs size={} type_bits=0x{:x}",
                dpb_mem_reqs.size, dpb_mem_reqs.memory_type_bits
            );
        }
        let dpb_type = pick_memory_type(
            &mem_props,
            dpb_mem_reqs.memory_type_bits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        .ok_or_else(|| Error::other("vulkan-video: no device-local memory type for DPB"))?;
        let dpb_alloc_info = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: dpb_mem_reqs.size,
            memory_type_index: dpb_type,
        };
        let mut dpb_memory: VkDeviceMemory = ptr::null_mut();
        let r = unsafe {
            (device.fns().allocate_memory)(
                device.handle(),
                &dpb_alloc_info,
                ptr::null(),
                &mut dpb_memory,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkAllocateMemory(DPB)", r));
        }
        let r =
            unsafe { (device.fns().bind_image_memory)(device.handle(), dpb_image, dpb_memory, 0) };
        if r != VK_SUCCESS {
            return Err(vk_err("vkBindImageMemory(DPB)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: DPB image+memory bound");
        }

        // DPB image view — base layer 0 (our setup slot is always 0).
        let dpb_view_ci = VkImageViewCreateInfo {
            s_type: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            image: dpb_image,
            view_type: VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            format: VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
            components: VkComponentMapping::default(),
            subresource_range: VkImageSubresourceRange {
                aspect_mask: VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: dpb_layers,
            },
        };
        let mut dpb_image_view: VkImageView = ptr::null_mut();
        let r = unsafe {
            (device.fns().create_image_view)(
                device.handle(),
                &dpb_view_ci,
                ptr::null(),
                &mut dpb_image_view,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkCreateImageView(DPB)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: DPB image view OK");
        }

        // ── Output image: same as DPB on coincide drivers ───────
        let (output_image, output_memory, output_image_view) = if coincide {
            // Reuse — but we need a separate single-layer view for the
            // dst_picture_resource (NVIDIA accepts the array view though).
            let output_view_ci = VkImageViewCreateInfo {
                s_type: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                image: dpb_image,
                view_type: VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                format: VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                components: VkComponentMapping::default(),
                subresource_range: VkImageSubresourceRange {
                    aspect_mask: VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: dpb_layers,
                },
            };
            let mut v: VkImageView = ptr::null_mut();
            let r = unsafe {
                (device.fns().create_image_view)(
                    device.handle(),
                    &output_view_ci,
                    ptr::null(),
                    &mut v,
                )
            };
            if r != VK_SUCCESS {
                return Err(vk_err("vkCreateImageView(output coincide)", r));
            }
            (dpb_image, dpb_memory, v)
        } else {
            let out_image_ci = VkImageCreateInfo {
                s_type: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                p_next: &profile_list as *const _ as *const c_void,
                flags: 0,
                image_type: VK_IMAGE_TYPE_2D,
                format: VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                extent: VkExtent3D {
                    width: max_w,
                    height: max_h,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: VK_SAMPLE_COUNT_1_BIT,
                tiling: VK_IMAGE_TILING_OPTIMAL,
                usage: VK_IMAGE_USAGE_VIDEO_DECODE_DST_BIT_KHR | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
                queue_family_index_count: 0,
                p_queue_family_indices: ptr::null(),
                initial_layout: VK_IMAGE_LAYOUT_UNDEFINED,
            };
            let mut img: VkImage = ptr::null_mut();
            let r = unsafe {
                (device.fns().create_image)(device.handle(), &out_image_ci, ptr::null(), &mut img)
            };
            if r != VK_SUCCESS {
                return Err(vk_err("vkCreateImage(output)", r));
            }
            let mut req = VkMemoryRequirements::default();
            unsafe {
                (device.fns().get_image_memory_requirements)(device.handle(), img, &mut req);
            }
            let t = pick_memory_type(
                &mem_props,
                req.memory_type_bits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            )
            .ok_or_else(|| Error::other("vulkan-video: no device-local for output image"))?;
            let alloc = VkMemoryAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                p_next: ptr::null(),
                allocation_size: req.size,
                memory_type_index: t,
            };
            let mut m: VkDeviceMemory = ptr::null_mut();
            let r = unsafe {
                (device.fns().allocate_memory)(device.handle(), &alloc, ptr::null(), &mut m)
            };
            if r != VK_SUCCESS {
                return Err(vk_err("vkAllocateMemory(output)", r));
            }
            let r = unsafe { (device.fns().bind_image_memory)(device.handle(), img, m, 0) };
            if r != VK_SUCCESS {
                return Err(vk_err("vkBindImageMemory(output)", r));
            }
            let view_ci = VkImageViewCreateInfo {
                s_type: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                p_next: ptr::null(),
                flags: 0,
                image: img,
                view_type: VK_IMAGE_VIEW_TYPE_2D,
                format: VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
                components: VkComponentMapping::default(),
                subresource_range: VkImageSubresourceRange {
                    aspect_mask: VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            let mut v: VkImageView = ptr::null_mut();
            let r = unsafe {
                (device.fns().create_image_view)(device.handle(), &view_ci, ptr::null(), &mut v)
            };
            if r != VK_SUCCESS {
                return Err(vk_err("vkCreateImageView(output)", r));
            }
            (img, m, v)
        };

        // ── Bitstream buffer (host-visible) ─────────────────────
        // Generous size — the full Annex-B picture for a small fixture
        // is well under 64 KiB. Caller can re-create on resize, this is
        // a Round-4 simplification.
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: creating bitstream buffer (coincide={})",
                coincide
            );
        }
        let bitstream_size = align_up(65536, caps.min_bitstream_buffer_offset_alignment.max(1));
        let buffer_ci = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: &profile_list as *const _ as *const c_void,
            flags: 0,
            size: bitstream_size,
            usage: VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };
        let mut bitstream_buffer: VkBuffer = ptr::null_mut();
        let r = unsafe {
            (device.fns().create_buffer)(
                device.handle(),
                &buffer_ci,
                ptr::null(),
                &mut bitstream_buffer,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkCreateBuffer(bitstream)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: bitstream buffer OK");
        }
        let mut bitstream_req = VkMemoryRequirements::default();
        unsafe {
            (device.fns().get_buffer_memory_requirements)(
                device.handle(),
                bitstream_buffer,
                &mut bitstream_req,
            );
        }
        let bitstream_t = pick_memory_type(
            &mem_props,
            bitstream_req.memory_type_bits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        .ok_or_else(|| Error::other("vulkan-video: no host-coherent memory for bitstream"))?;
        let alloc = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: bitstream_req.size,
            memory_type_index: bitstream_t,
        };
        let mut bitstream_memory: VkDeviceMemory = ptr::null_mut();
        let r = unsafe {
            (device.fns().allocate_memory)(
                device.handle(),
                &alloc,
                ptr::null(),
                &mut bitstream_memory,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkAllocateMemory(bitstream)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: bitstream memory allocated");
        }
        let r = unsafe {
            (device.fns().bind_buffer_memory)(
                device.handle(),
                bitstream_buffer,
                bitstream_memory,
                0,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkBindBufferMemory(bitstream)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: bitstream memory bound");
        }

        // ── Staging buffer (host-visible, big enough for NV12) ──
        let luma_stride = max_w;
        let chroma_stride = max_w; // NV12: interleaved UV pair = 2 bytes per chroma sample, but stride width maps as luma width
        let chroma_height = max_h.div_ceil(2);
        let staging_size = (luma_stride * max_h + chroma_stride * chroma_height) as u64;
        let staging_size = staging_size.max(1024);

        let staging_ci = VkBufferCreateInfo {
            s_type: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            size: staging_size,
            usage: VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharing_mode: VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: creating staging buffer (size={})",
                staging_size
            );
        }
        let mut staging_buffer: VkBuffer = ptr::null_mut();
        let r = unsafe {
            (device.fns().create_buffer)(
                device.handle(),
                &staging_ci,
                ptr::null(),
                &mut staging_buffer,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkCreateBuffer(staging)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: staging buffer OK");
        }
        let mut staging_req = VkMemoryRequirements::default();
        unsafe {
            (device.fns().get_buffer_memory_requirements)(
                device.handle(),
                staging_buffer,
                &mut staging_req,
            );
        }
        let staging_t = pick_memory_type(
            &mem_props,
            staging_req.memory_type_bits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        .ok_or_else(|| Error::other("vulkan-video: no host-coherent memory for staging"))?;
        let alloc = VkMemoryAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: staging_req.size,
            memory_type_index: staging_t,
        };
        let mut staging_memory: VkDeviceMemory = ptr::null_mut();
        let r = unsafe {
            (device.fns().allocate_memory)(
                device.handle(),
                &alloc,
                ptr::null(),
                &mut staging_memory,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkAllocateMemory(staging)", r));
        }
        let r = unsafe {
            (device.fns().bind_buffer_memory)(device.handle(), staging_buffer, staging_memory, 0)
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkBindBufferMemory(staging)", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: staging buffer bound");
        }

        // ── Command pool + command buffer ──────────────────────
        let cp_ci = VkCommandPoolCreateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_index: qfi,
        };
        let mut command_pool: VkCommandPool = ptr::null_mut();
        let r = unsafe {
            (device.fns().create_command_pool)(
                device.handle(),
                &cp_ci,
                ptr::null(),
                &mut command_pool,
            )
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkCreateCommandPool", r));
        }
        let cb_ai = VkCommandBufferAllocateInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut command_buffer: VkCommandBuffer = ptr::null_mut();
        let r = unsafe {
            (device.fns().allocate_command_buffers)(device.handle(), &cb_ai, &mut command_buffer)
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkAllocateCommandBuffers", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: command pool + buffer OK");
        }

        Ok(Self {
            command_buffer,
            command_pool,
            bitstream_buffer,
            bitstream_memory,
            bitstream_size,
            staging_buffer,
            staging_memory,
            staging_size,
            output_image_view,
            output_image,
            output_memory,
            dpb_image_view,
            dpb_image,
            dpb_memory,
            session_params,
            // SAFETY: extends the `'device` borrow on `video_session` to
            // `'static`; the session is owned by `Self` and dropped before
            // the `Device` it borrows, so the apparent lifetime is sound.
            session: Some(unsafe {
                std::mem::transmute::<VideoSession<'_>, VideoSession<'static>>(video_session)
            }),
            queue_family_index: qfi,
            physical_device_handle: pd_handle,
            device,
            instance,
            width: sps.display_width().min(max_w),
            height: sps.display_height().min(max_h),
            luma_stride,
            chroma_stride,
            chroma_height,
            coincide,
            bitstream_offset_alignment: caps.min_bitstream_buffer_offset_alignment.max(1),
            bitstream_size_alignment: caps.min_bitstream_buffer_size_alignment.max(1),
        })
    }

    fn decode_picture(
        &mut self,
        bitstream: &[u8],
        sps: &H264Sps,
        out_queue: &mut VecDeque<VideoFrame>,
    ) -> Result<()> {
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: decode_picture (bitstream={})",
                bitstream.len()
            );
        }
        // ── Upload bitstream to host-visible buffer ────────────
        if (bitstream.len() as u64) > self.bitstream_size {
            return Err(Error::other(format!(
                "vulkan-video: bitstream {} > buffer {}",
                bitstream.len(),
                self.bitstream_size
            )));
        }
        unsafe {
            let mut p: *mut c_void = ptr::null_mut();
            let r = (self.device.fns().map_memory)(
                self.device.handle(),
                self.bitstream_memory,
                0,
                self.bitstream_size,
                0,
                &mut p,
            );
            if r != VK_SUCCESS {
                return Err(vk_err("vkMapMemory(bitstream)", r));
            }
            if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
                eprintln!("vulkan-video: bitstream mapped at {:p}", p);
            }
            // Zero-fill remainder, then copy.
            std::ptr::write_bytes(p as *mut u8, 0, self.bitstream_size as usize);
            std::ptr::copy_nonoverlapping(bitstream.as_ptr(), p as *mut u8, bitstream.len());
            (self.device.fns().unmap_memory)(self.device.handle(), self.bitstream_memory);
            if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
                eprintln!("vulkan-video: bitstream uploaded");
            }
        }

        // ── Build the various structs for the command buffer ───
        let std_pic_info = StdVideoDecodeH264PictureInfo {
            flags: StdVideoDecodeH264PictureInfoFlags {
                flags: StdVideoDecodeH264PictureInfoFlags::IS_INTRA
                    | StdVideoDecodeH264PictureInfoFlags::IDR_PIC
                    | StdVideoDecodeH264PictureInfoFlags::IS_REFERENCE,
            },
            seq_parameter_set_id: sps.seq_parameter_set_id,
            pic_parameter_set_id: 0,
            reserved1: 0,
            reserved2: 0,
            frame_num: 0,
            idr_pic_id: 0,
            pic_order_cnt: [0, 0],
        };

        // Find each VCL slice NAL and emit a slice offset that
        // points to the start code prefix of that NAL, relative to
        // srcBufferOffset (= 0 in our case).
        let slice_offsets = compute_slice_offsets(bitstream);
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: {} VCL slices at offsets {:?}",
                slice_offsets.len(),
                slice_offsets
            );
        }
        let slice_count = slice_offsets.len() as u32;
        let h264_pic_info = VkVideoDecodeH264PictureInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PICTURE_INFO_KHR,
            p_next: ptr::null(),
            p_std_picture_info: &std_pic_info,
            slice_count,
            p_slice_offsets: if slice_offsets.is_empty() {
                ptr::null()
            } else {
                slice_offsets.as_ptr()
            },
        };

        // DPB reference slot for the IDR's reconstructed picture.
        let dpb_picture_resource = VkVideoPictureResourceInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
            p_next: ptr::null(),
            coded_offset: VkOffset2D { x: 0, y: 0 },
            coded_extent: VkExtent2D {
                width: sps.coded_width().max(self.width),
                height: sps.coded_height().max(self.height),
            },
            base_array_layer: 0,
            image_view_binding: self.dpb_image_view,
        };

        let std_ref_info = StdVideoDecodeH264ReferenceInfo::default();
        let h264_dpb_slot = VkVideoDecodeH264DpbSlotInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_DPB_SLOT_INFO_KHR,
            p_next: ptr::null(),
            p_std_reference_info: &std_ref_info,
        };

        // Setup-reference slot describing the IDR's reconstructed picture (slot 0).
        let setup_ref = VkVideoReferenceSlotInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
            p_next: &h264_dpb_slot as *const _ as *const c_void,
            slot_index: 0,
            p_picture_resource: &dpb_picture_resource,
        };

        // For an IDR with no priors, the active reference list is
        // empty. The setup-reference slot declared on the decode info
        // describes the slot the IDR's reconstruction will occupy.
        // We list that slot here too with slot_index=-1 (inactive)
        // so the driver knows to allocate it.
        let begin_ref = VkVideoReferenceSlotInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
            p_next: ptr::null(),
            slot_index: -1,
            p_picture_resource: &dpb_picture_resource,
        };

        // dst_picture_resource — where the GPU writes the decoded
        // output. On coincide drivers this is the same view+layer as
        // the setup reference (`output_image_view` aliases the DPB
        // image); on distinct, it's the separate output image. Either
        // way `output_image_view` was constructed to point at the
        // correct backing image, so the descriptor is identical.
        let dst_picture_resource = VkVideoPictureResourceInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
            p_next: ptr::null(),
            coded_offset: VkOffset2D { x: 0, y: 0 },
            coded_extent: VkExtent2D {
                width: sps.coded_width().max(self.width),
                height: sps.coded_height().max(self.height),
            },
            base_array_layer: 0,
            image_view_binding: self.output_image_view,
        };

        let session_handle = self
            .session
            .as_ref()
            .ok_or_else(|| Error::other("vulkan-video: no session"))?
            .handle();

        let begin_info = VkVideoBeginCodingInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
            p_next: ptr::null(),
            flags: 0,
            video_session: session_handle,
            video_session_parameters: self.session_params,
            reference_slot_count: 1,
            p_reference_slots: &begin_ref,
        };

        let control_info = VkVideoCodingControlInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
            p_next: ptr::null(),
            flags: VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR,
        };

        let decode_info = VkVideoDecodeInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_INFO_KHR,
            p_next: &h264_pic_info as *const _ as *const c_void,
            flags: 0,
            src_buffer: self.bitstream_buffer,
            src_buffer_offset: 0,
            src_buffer_range: align_up(bitstream.len() as u64, self.bitstream_size_alignment),
            dst_picture_resource,
            p_setup_reference_slot: &setup_ref,
            reference_slot_count: 0,
            p_reference_slots: ptr::null(),
        };

        let end_info = VkVideoEndCodingInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR,
            p_next: ptr::null(),
            flags: 0,
        };

        // ── Record command buffer ──────────────────────────────
        let cb_begin = VkCommandBufferBeginInfo {
            s_type: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: ptr::null(),
        };
        let r = unsafe { (self.device.fns().begin_command_buffer)(self.command_buffer, &cb_begin) };
        if r != VK_SUCCESS {
            return Err(vk_err("vkBeginCommandBuffer", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: command buffer recording started");
        }

        // Transition DPB image (and output image, if distinct) from
        // UNDEFINED to VIDEO_DECODE_DPB_KHR / DECODE_DST_KHR.
        let mut barriers: Vec<VkImageMemoryBarrier> = Vec::new();
        barriers.push(VkImageMemoryBarrier {
            s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: 0,
            dst_access_mask: VK_ACCESS_MEMORY_WRITE_BIT,
            old_layout: VK_IMAGE_LAYOUT_UNDEFINED,
            new_layout: if self.coincide {
                VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR
            } else {
                VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR
            },
            src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
            image: self.dpb_image,
            subresource_range: VkImageSubresourceRange {
                aspect_mask: VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        });
        if !self.coincide {
            barriers.push(VkImageMemoryBarrier {
                s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: ptr::null(),
                src_access_mask: 0,
                dst_access_mask: VK_ACCESS_MEMORY_WRITE_BIT,
                old_layout: VK_IMAGE_LAYOUT_UNDEFINED,
                new_layout: VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
                src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
                image: self.output_image,
                subresource_range: VkImageSubresourceRange {
                    aspect_mask: VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            });
        }
        unsafe {
            (self.device.fns().cmd_pipeline_barrier)(
                self.command_buffer,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0,
                0,
                ptr::null(),
                0,
                ptr::null(),
                barriers.len() as u32,
                barriers.as_ptr(),
            );
        }

        // Begin video coding scope.
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!(
                "vulkan-video: recording video coding commands (skip_decode={})",
                std::env::var("OXIDEAV_VK_SKIP_DECODE").is_ok()
            );
        }
        let skip_decode = std::env::var("OXIDEAV_VK_SKIP_DECODE").is_ok();
        unsafe {
            (self.device.fns().cmd_begin_video_coding_khr)(self.command_buffer, &begin_info);
            if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
                eprintln!("vulkan-video: cmd_begin_video_coding done");
            }
            (self.device.fns().cmd_control_video_coding_khr)(self.command_buffer, &control_info);
            if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
                eprintln!("vulkan-video: cmd_control done");
            }
            if !skip_decode {
                (self.device.fns().cmd_decode_video_khr)(self.command_buffer, &decode_info);
                if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
                    eprintln!("vulkan-video: cmd_decode done");
                }
            }
            (self.device.fns().cmd_end_video_coding_khr)(self.command_buffer, &end_info);
            if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
                eprintln!("vulkan-video: cmd_end done");
            }
        }

        // Transition output image to TRANSFER_SRC_OPTIMAL. After the
        // decode submission the output image (whether aliased onto the
        // DPB on coincide drivers or a distinct image otherwise) is in
        // VIDEO_DECODE_DST_KHR — see the pre-decode barrier above.
        let src_layout = VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR;
        let to_xfer = VkImageMemoryBarrier {
            s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: VK_ACCESS_MEMORY_WRITE_BIT,
            dst_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
            old_layout: src_layout,
            new_layout: VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
            image: self.output_image,
            subresource_range: VkImageSubresourceRange {
                aspect_mask: VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        };
        unsafe {
            (self.device.fns().cmd_pipeline_barrier)(
                self.command_buffer,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                ptr::null(),
                0,
                ptr::null(),
                1,
                &to_xfer,
            );
        }

        // Copy luma + chroma planes out via two regions.
        let luma_bytes = (self.luma_stride as u64) * (self.height as u64);
        let _luma_bytes_aligned = align_up(luma_bytes, 16);
        let regions = [
            VkBufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: self.luma_stride,
                buffer_image_height: self.height,
                image_subresource: VkImageSubresourceLayers {
                    aspect_mask: VK_IMAGE_ASPECT_PLANE_0_BIT,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: VkOffset3D::default(),
                image_extent: VkExtent3D {
                    width: self.width,
                    height: self.height,
                    depth: 1,
                },
            },
            VkBufferImageCopy {
                buffer_offset: luma_bytes,
                buffer_row_length: self.chroma_stride,
                buffer_image_height: self.chroma_height,
                image_subresource: VkImageSubresourceLayers {
                    aspect_mask: VK_IMAGE_ASPECT_PLANE_1_BIT,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: VkOffset3D::default(),
                image_extent: VkExtent3D {
                    width: self.width.div_ceil(2),
                    height: self.chroma_height,
                    depth: 1,
                },
            },
        ];
        unsafe {
            (self.device.fns().cmd_copy_image_to_buffer)(
                self.command_buffer,
                self.output_image,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                self.staging_buffer,
                regions.len() as u32,
                regions.as_ptr(),
            );
        }

        // Transition output back to a quiescent state for the next decode.
        let to_quiescent = VkImageMemoryBarrier {
            s_type: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            p_next: ptr::null(),
            src_access_mask: VK_ACCESS_TRANSFER_READ_BIT,
            dst_access_mask: VK_ACCESS_MEMORY_READ_BIT,
            old_layout: VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            new_layout: VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR,
            src_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: VK_QUEUE_FAMILY_IGNORED,
            image: self.output_image,
            subresource_range: VkImageSubresourceRange {
                aspect_mask: VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
        };
        unsafe {
            (self.device.fns().cmd_pipeline_barrier)(
                self.command_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0,
                0,
                ptr::null(),
                0,
                ptr::null(),
                1,
                &to_quiescent,
            );
        }

        let r = unsafe { (self.device.fns().end_command_buffer)(self.command_buffer) };
        if r != VK_SUCCESS {
            return Err(vk_err("vkEndCommandBuffer", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: end_command_buffer OK, submitting");
        }

        if std::env::var("OXIDEAV_VK_SKIP_SUBMIT").is_ok() {
            return Err(Error::other("OXIDEAV_VK_SKIP_SUBMIT set; skipping submit"));
        }

        // ── Submit ─────────────────────────────────────────────
        let queue = self.device.queue(self.queue_family_index);
        let submit = VkSubmitInfo {
            s_type: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffer,
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        };
        let r = unsafe {
            (self.device.fns().queue_submit)(queue.handle(), 1, &submit, ptr::null_mut())
        };
        if r != VK_SUCCESS {
            return Err(vk_err("vkQueueSubmit", r));
        }
        let r = unsafe { (self.device.fns().queue_wait_idle)(queue.handle()) };
        if r != VK_SUCCESS {
            return Err(vk_err("vkQueueWaitIdle", r));
        }
        if std::env::var("OXIDEAV_VK_TRACE").is_ok() {
            eprintln!("vulkan-video: GPU done, reading staging");
        }

        // ── Read pixels back from staging ───────────────────────
        let mut frame_y = vec![0u8; (self.width as usize) * (self.height as usize)];
        let cw = self.width.div_ceil(2) as usize;
        let ch = self.chroma_height as usize;
        let mut frame_u = vec![0u8; cw * ch];
        let mut frame_v = vec![0u8; cw * ch];

        unsafe {
            let mut p: *mut c_void = ptr::null_mut();
            let r = (self.device.fns().map_memory)(
                self.device.handle(),
                self.staging_memory,
                0,
                self.staging_size,
                0,
                &mut p,
            );
            if r != VK_SUCCESS {
                return Err(vk_err("vkMapMemory(staging)", r));
            }
            let host = p as *const u8;
            // Copy luma row-by-row (stride = luma_stride, dst stride = width).
            let lstride = self.luma_stride as usize;
            for y in 0..self.height as usize {
                let src = host.add(y * lstride);
                let dst = frame_y.as_mut_ptr().add(y * self.width as usize);
                std::ptr::copy_nonoverlapping(src, dst, self.width as usize);
            }
            // De-interleave NV12 chroma: UV-pair-per-2x2-luma-block.
            let chroma_off = (self.luma_stride as usize) * (self.height as usize);
            let cstride = self.chroma_stride as usize;
            for y in 0..ch {
                for x in 0..cw {
                    let src_pair = host.add(chroma_off + y * cstride + x * 2);
                    let u = *src_pair;
                    let v = *src_pair.add(1);
                    frame_u[y * cw + x] = u;
                    frame_v[y * cw + x] = v;
                }
            }
            (self.device.fns().unmap_memory)(self.device.handle(), self.staging_memory);
        }

        out_queue.push_back(VideoFrame {
            pts: None,
            planes: vec![
                VideoPlane {
                    stride: self.width as usize,
                    data: frame_y,
                },
                VideoPlane {
                    stride: cw,
                    data: frame_u,
                },
                VideoPlane {
                    stride: cw,
                    data: frame_v,
                },
            ],
        });
        Ok(())
    }
}

// ─────────────────────── H264Sps/H264Pps → Std structs ──────────────────────

fn std_sps_from_parsed(s: &H264Sps) -> StdVideoH264SequenceParameterSet {
    let mut flags: u32 = 0;
    if s.constraint_set_flags & 0x80 != 0 {
        flags |= StdVideoH264SpsFlags::CONSTRAINT_SET0;
    }
    if s.constraint_set_flags & 0x40 != 0 {
        flags |= StdVideoH264SpsFlags::CONSTRAINT_SET1;
    }
    if s.direct_8x8_inference_flag {
        flags |= StdVideoH264SpsFlags::DIRECT_8X8_INFERENCE;
    }
    if s.mb_adaptive_frame_field_flag {
        flags |= StdVideoH264SpsFlags::MB_ADAPTIVE_FRAME_FIELD;
    }
    if s.frame_mbs_only_flag {
        flags |= StdVideoH264SpsFlags::FRAME_MBS_ONLY;
    }
    if s.gaps_in_frame_num_value_allowed_flag {
        flags |= StdVideoH264SpsFlags::GAPS_IN_FRAME_NUM;
    }
    let crop = s.frame_cropping.unwrap_or_default();
    if s.frame_cropping.is_some() {
        flags |= StdVideoH264SpsFlags::FRAME_CROPPING;
    }
    // VUI is not parsed by oxideav-bitstream, so we never set the
    // VUI_PARAMETERS_PRESENT flag and leave `p_sequence_parameter_set_vui`
    // null. The IDR-only fixture works fine without it.
    StdVideoH264SequenceParameterSet {
        flags: StdVideoH264SpsFlags { flags },
        profile_idc: s.profile_idc as i32,
        level_idc: h264_level_byte_to_idc(s.level_idc),
        chroma_format_idc: STD_VIDEO_H264_CHROMA_FORMAT_IDC_420,
        seq_parameter_set_id: s.seq_parameter_set_id,
        bit_depth_luma_minus8: s.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: s.bit_depth_chroma_minus8,
        log2_max_frame_num_minus4: s.log2_max_frame_num_minus4,
        pic_order_cnt_type: s.pic_order_cnt_type as i32,
        offset_for_non_ref_pic: 0,
        offset_for_top_to_bottom_field: 0,
        log2_max_pic_order_cnt_lsb_minus4: s.log2_max_pic_order_cnt_lsb_minus4,
        num_ref_frames_in_pic_order_cnt_cycle: 0,
        max_num_ref_frames: s.max_num_ref_frames as u8,
        reserved1: 0,
        pic_width_in_mbs_minus1: s.pic_width_in_mbs_minus1,
        pic_height_in_map_units_minus1: s.pic_height_in_map_units_minus1,
        frame_crop_left_offset: crop.left,
        frame_crop_right_offset: crop.right,
        frame_crop_top_offset: crop.top,
        frame_crop_bottom_offset: crop.bottom,
        reserved2: 0,
        p_offset_for_ref_frame: ptr::null(),
        p_scaling_lists: ptr::null(),
        p_sequence_parameter_set_vui: ptr::null(),
    }
}

fn std_pps_from_parsed(p: &H264Pps) -> StdVideoH264PictureParameterSet {
    let mut flags: u32 = 0;
    if p.transform_8x8_mode_flag {
        flags |= StdVideoH264PpsFlags::TRANSFORM_8X8_MODE;
    }
    if p.redundant_pic_cnt_present_flag {
        flags |= StdVideoH264PpsFlags::REDUNDANT_PIC_CNT;
    }
    if p.constrained_intra_pred_flag {
        flags |= StdVideoH264PpsFlags::CONSTRAINED_INTRA_PRED;
    }
    if p.deblocking_filter_control_present_flag {
        flags |= StdVideoH264PpsFlags::DEBLOCK_FILTER_CTRL;
    }
    if p.weighted_pred_flag {
        flags |= StdVideoH264PpsFlags::WEIGHTED_PRED;
    }
    if p.bottom_field_pic_order_in_frame_present_flag {
        flags |= StdVideoH264PpsFlags::BOTTOM_FIELD_POC_IN_FRAME;
    }
    if p.entropy_coding_mode_flag {
        flags |= StdVideoH264PpsFlags::ENTROPY_CODING_MODE;
    }
    // pic_scaling_matrix_present_flag — oxideav-bitstream rejects PPS
    // with scaling-matrix-present (Unsupported) so by construction this
    // flag is always 0 for any parsed PPS we get here.
    StdVideoH264PictureParameterSet {
        flags: StdVideoH264PpsFlags { flags },
        seq_parameter_set_id: p.seq_parameter_set_id,
        pic_parameter_set_id: p.pic_parameter_set_id,
        num_ref_idx_l0_default_active_minus1: p.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1: p.num_ref_idx_l1_default_active_minus1,
        weighted_bipred_idc: p.weighted_bipred_idc as i32,
        pic_init_qp_minus26: p.pic_init_qp_minus26 as i8,
        pic_init_qs_minus26: p.pic_init_qs_minus26 as i8,
        chroma_qp_index_offset: p.chroma_qp_index_offset as i8,
        second_chroma_qp_index_offset: p.second_chroma_qp_index_offset as i8,
        p_scaling_lists: ptr::null(),
    }
}

/// Convert the H.264 raw `level_idc` byte (which encodes 30, 31, 41…
/// in the bitstream) to the contiguous `StdVideoH264LevelIdc` enum
/// value (1.0 → 0, 1.1 → 1, …, 5.1 → 14, …, 6.2 → 18).
fn h264_level_byte_to_idc(b: u8) -> sys::StdVideoH264LevelIdc {
    match b {
        10 => 0,  // 1.0
        11 => 1,  // 1.1
        12 => 2,  // 1.2
        13 => 3,  // 1.3
        20 => 4,  // 2.0
        21 => 5,  // 2.1
        22 => 6,  // 2.2
        30 => 7,  // 3.0
        31 => 8,  // 3.1
        32 => 9,  // 3.2
        40 => 10, // 4.0
        41 => 11, // 4.1
        42 => 12, // 4.2
        50 => 13, // 5.0
        51 => 14, // 5.1
        52 => 15, // 5.2
        60 => 16, // 6.0
        61 => 17, // 6.1
        62 => 18, // 6.2
        _ => 14,  // default to 5.1
    }
}
