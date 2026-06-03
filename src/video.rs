//! Video session wrapper + capability queries.
//!
//! Round 3 surface:
//!
//! * [`query_video_decode_h264_capabilities`] — instance-level
//!   `vkGetPhysicalDeviceVideoCapabilitiesKHR` call wired through the
//!   chained `VkVideoProfileInfoKHR` → `VkVideoDecodeH264ProfileInfoKHR`
//!   pNext path, with a chained `VkVideoCapabilitiesKHR` →
//!   `VkVideoDecodeCapabilitiesKHR` → `VkVideoDecodeH264CapabilitiesKHR`
//!   output chain.
//! * [`VideoSession`] — RAII wrapper around `VkVideoSessionKHR`
//!   created via `vkCreateVideoSessionKHR`. Keeps a reference to the
//!   parent [`Device`] for lifetime tracking and access to the
//!   device dispatch table on Drop.
//! * [`VideoSession::memory_requirements`] — calls
//!   `vkGetVideoSessionMemoryRequirementsKHR` and returns a `Vec` of
//!   `VkVideoSessionMemoryRequirementsKHR` records.
//! * [`VideoSession::allocate_and_bind_memory`] — allocates
//!   `VkDeviceMemory` for each requirement and calls
//!   `vkBindVideoSessionMemoryKHR`. The owned allocations are
//!   tracked on the session and freed in `Drop`.

use std::ffi::c_void;
use std::os::raw::c_char;
use std::ptr;

use crate::device::Device;
use crate::instance::VkError;
use crate::physical_device::PhysicalDevice;
use crate::sys::{
    self, StdVideoAV1Level, StdVideoAV1Profile, StdVideoH264LevelIdc, StdVideoH264ProfileIdc,
    StdVideoH265LevelIdc, StdVideoH265ProfileIdc, VkBindVideoSessionMemoryInfoKHR, VkDeviceMemory,
    VkExtensionProperties, VkExtent2D, VkMemoryAllocateInfo, VkMemoryRequirements,
    VkPhysicalDeviceMemoryProperties, VkVideoCapabilitiesKHR, VkVideoDecodeAV1CapabilitiesKHR,
    VkVideoDecodeAV1ProfileInfoKHR, VkVideoDecodeCapabilitiesKHR, VkVideoDecodeH264CapabilitiesKHR,
    VkVideoDecodeH264ProfileInfoKHR, VkVideoDecodeH265CapabilitiesKHR,
    VkVideoDecodeH265ProfileInfoKHR, VkVideoProfileInfoKHR, VkVideoSessionCreateInfoKHR,
    VkVideoSessionKHR, VkVideoSessionMemoryRequirementsKHR, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
    VK_MAX_EXTENSION_NAME_SIZE, VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_EXTENSION_NAME,
    VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_SPEC_VERSION,
    VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR, VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR, VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR,
    VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR, VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
    VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR, VK_SUCCESS,
    VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR, VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR,
    VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR, VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
    VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR, VK_VIDEO_DECODE_H264_PICTURE_LAYOUT_PROGRESSIVE_KHR,
};

// ─────────────────────────── VideoCapabilities ───────────────────────────────

/// Decoded form of the `VkVideoCapabilitiesKHR` chain returned by
/// `vkGetPhysicalDeviceVideoCapabilitiesKHR` for an H.264 decode
/// profile. Only the fields the rest of oxideav cares about are kept;
/// the raw chain is dropped after the call.
#[derive(Debug, Clone)]
pub struct VideoDecodeH264Capabilities {
    /// Smallest image extent the decoder accepts (typically 16x16).
    pub min_coded_extent: (u32, u32),
    /// Largest image extent the decoder will produce.
    pub max_coded_extent: (u32, u32),
    /// Granularity (in pixels) at which `decode_target` regions must
    /// be aligned.
    pub picture_access_granularity: (u32, u32),
    /// Maximum DPB slot count the implementation will support for the
    /// queried profile (this is what bounds the H.264 reference list).
    pub max_dpb_slots: u32,
    /// Maximum count of active reference pictures simultaneously
    /// drawable from the DPB.
    pub max_active_reference_pictures: u32,
    /// Maximum H.264 level the implementation supports for the
    /// queried profile, as a `StdVideoH264LevelIdc`.
    pub max_level_idc: StdVideoH264LevelIdc,
    /// Implementation-supplied `VkExtensionProperties` describing the
    /// `VK_STD_vulkan_video_codec_h264_decode` header version the
    /// driver is implemented against. Round 3 echoes this value
    /// straight into `VkVideoSessionCreateInfoKHR.pStdHeaderVersion`.
    pub std_header_version: VkExtensionProperties,
    /// Capability flags from `VkVideoCapabilitiesKHR.flags` (raw bits).
    pub capability_flags: u32,
    /// Decode-specific capability flags from
    /// `VkVideoDecodeCapabilitiesKHR.flags` (raw bits).
    pub decode_capability_flags: u32,
    /// Bitstream-buffer offset alignment from
    /// `VkVideoCapabilitiesKHR.min_bitstream_buffer_offset_alignment`.
    pub min_bitstream_buffer_offset_alignment: u64,
    /// Bitstream-buffer size alignment.
    pub min_bitstream_buffer_size_alignment: u64,
}

/// Run `vkGetPhysicalDeviceVideoCapabilitiesKHR` for an H.264 decode
/// profile.
///
/// `profile_idc` is one of the `STD_VIDEO_H264_PROFILE_IDC_*`
/// constants in [`crate::sys`] (e.g. `STD_VIDEO_H264_PROFILE_IDC_HIGH`).
/// The function builds the mandatory chained
/// `VkVideoProfileInfoKHR` → `VkVideoDecodeH264ProfileInfoKHR` and a
/// matching `VkVideoCapabilitiesKHR` → `VkVideoDecodeCapabilitiesKHR`
/// → `VkVideoDecodeH264CapabilitiesKHR` output chain.
pub fn query_video_decode_h264_capabilities(
    physical_device: &PhysicalDevice<'_>,
    profile_idc: StdVideoH264ProfileIdc,
) -> Result<VideoDecodeH264Capabilities, VkError> {
    let fns = physical_device.instance_fns();
    let get_caps =
        fns.get_physical_device_video_capabilities_khr
            .ok_or(VkError::MissingFunction(
                "vkGetPhysicalDeviceVideoCapabilitiesKHR",
            ))?;

    // ─ Profile chain ────────────────────────────────────────────
    let h264_profile = VkVideoDecodeH264ProfileInfoKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,
        p_next: ptr::null(),
        std_profile_idc: profile_idc,
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

    // ─ Capabilities chain (output) ─────────────────────────────
    // Initialise the inner-most struct first so its address is
    // stable; chain back outwards. SAFETY: each struct's `s_type` is
    // set by us before the call; Vulkan writes the rest.
    let mut h264_caps = VkVideoDecodeH264CapabilitiesKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_CAPABILITIES_KHR,
        p_next: ptr::null_mut(),
        max_level_idc: 0,
        field_offset_granularity: sys::VkOffset2D::default(),
    };
    let mut decode_caps = VkVideoDecodeCapabilitiesKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,
        p_next: &mut h264_caps as *mut _ as *mut c_void,
        flags: 0,
    };
    let mut caps = VkVideoCapabilitiesKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
        p_next: &mut decode_caps as *mut _ as *mut c_void,
        flags: 0,
        min_bitstream_buffer_offset_alignment: 0,
        min_bitstream_buffer_size_alignment: 0,
        picture_access_granularity: VkExtent2D::default(),
        min_coded_extent: VkExtent2D::default(),
        max_coded_extent: VkExtent2D::default(),
        max_dpb_slots: 0,
        max_active_reference_pictures: 0,
        std_header_version: VkExtensionProperties {
            extension_name: [0; VK_MAX_EXTENSION_NAME_SIZE],
            spec_version: 0,
        },
    };

    // SAFETY: chain pointers reference local stack objects that live
    // until the call returns; each struct has its `sType`
    // discriminant set per the spec.
    let result = unsafe { get_caps(physical_device.handle(), &profile, &mut caps) };
    if result != VK_SUCCESS {
        return Err(VkError::Result {
            op: "vkGetPhysicalDeviceVideoCapabilitiesKHR",
            result,
        });
    }

    Ok(VideoDecodeH264Capabilities {
        min_coded_extent: (caps.min_coded_extent.width, caps.min_coded_extent.height),
        max_coded_extent: (caps.max_coded_extent.width, caps.max_coded_extent.height),
        picture_access_granularity: (
            caps.picture_access_granularity.width,
            caps.picture_access_granularity.height,
        ),
        max_dpb_slots: caps.max_dpb_slots,
        max_active_reference_pictures: caps.max_active_reference_pictures,
        max_level_idc: h264_caps.max_level_idc,
        std_header_version: caps.std_header_version,
        capability_flags: caps.flags,
        decode_capability_flags: decode_caps.flags,
        min_bitstream_buffer_offset_alignment: caps.min_bitstream_buffer_offset_alignment,
        min_bitstream_buffer_size_alignment: caps.min_bitstream_buffer_size_alignment,
    })
}

// ─────────────────────────── H.265 (HEVC) capabilities ───────────────────────

/// Decoded form of the H.265 decode capability chain returned by
/// `vkGetPhysicalDeviceVideoCapabilitiesKHR`. Shares the common
/// `VkVideoCapabilitiesKHR` fields with the H.264 variant and adds
/// `max_level_idc` from `VkVideoDecodeH265CapabilitiesKHR`.
#[derive(Debug, Clone)]
pub struct VideoDecodeH265Capabilities {
    /// Smallest image extent the decoder accepts (typically 16x16 or
    /// 8x8 for HEVC CTU-aligned).
    pub min_coded_extent: (u32, u32),
    /// Largest image extent the decoder will produce.
    pub max_coded_extent: (u32, u32),
    /// Granularity (in pixels) at which `decode_target` regions must
    /// be aligned.
    pub picture_access_granularity: (u32, u32),
    /// Maximum DPB slot count the implementation will support for the
    /// queried profile.
    pub max_dpb_slots: u32,
    /// Maximum count of active reference pictures simultaneously
    /// drawable from the DPB.
    pub max_active_reference_pictures: u32,
    /// Maximum H.265 level the implementation supports for the
    /// queried profile, as a `StdVideoH265LevelIdc` enum index.
    pub max_level_idc: StdVideoH265LevelIdc,
    /// Implementation-supplied `VkExtensionProperties` describing the
    /// `VK_STD_vulkan_video_codec_h265_decode` header version the
    /// driver is implemented against.
    pub std_header_version: VkExtensionProperties,
    /// Capability flags from `VkVideoCapabilitiesKHR.flags` (raw bits).
    pub capability_flags: u32,
    /// Decode-specific capability flags from
    /// `VkVideoDecodeCapabilitiesKHR.flags` (raw bits).
    pub decode_capability_flags: u32,
    /// Bitstream-buffer offset alignment.
    pub min_bitstream_buffer_offset_alignment: u64,
    /// Bitstream-buffer size alignment.
    pub min_bitstream_buffer_size_alignment: u64,
}

/// Run `vkGetPhysicalDeviceVideoCapabilitiesKHR` for an H.265 decode
/// profile.
///
/// `profile_idc` is one of the `STD_VIDEO_H265_PROFILE_IDC_*`
/// constants in [`crate::sys`] (e.g.
/// `STD_VIDEO_H265_PROFILE_IDC_MAIN`). The function builds the
/// mandatory chained `VkVideoProfileInfoKHR` →
/// `VkVideoDecodeH265ProfileInfoKHR` and a matching
/// `VkVideoCapabilitiesKHR` → `VkVideoDecodeCapabilitiesKHR` →
/// `VkVideoDecodeH265CapabilitiesKHR` output chain.
///
/// Main/Main-still-picture profiles take `8_BIT` luma/chroma. Main-10
/// requires the caller to swap the `*_BIT_DEPTH_*` arguments to a
/// 10-bit constant — that's not exposed yet (profile-driven bit
/// depth is a follow-up; the 8-bit Main case is what every consumer
/// codec entry needs for the `engine_info()` row).
pub fn query_video_decode_h265_capabilities(
    physical_device: &PhysicalDevice<'_>,
    profile_idc: StdVideoH265ProfileIdc,
) -> Result<VideoDecodeH265Capabilities, VkError> {
    let fns = physical_device.instance_fns();
    let get_caps =
        fns.get_physical_device_video_capabilities_khr
            .ok_or(VkError::MissingFunction(
                "vkGetPhysicalDeviceVideoCapabilitiesKHR",
            ))?;

    let h265_profile = VkVideoDecodeH265ProfileInfoKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_PROFILE_INFO_KHR,
        p_next: ptr::null(),
        std_profile_idc: profile_idc,
    };
    let profile = VkVideoProfileInfoKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
        p_next: &h265_profile as *const _ as *const c_void,
        video_codec_operation: VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
        chroma_subsampling: VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
        luma_bit_depth: VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
        chroma_bit_depth: VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    };

    let mut h265_caps = VkVideoDecodeH265CapabilitiesKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_CAPABILITIES_KHR,
        p_next: ptr::null_mut(),
        max_level_idc: 0,
    };
    let mut decode_caps = VkVideoDecodeCapabilitiesKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,
        p_next: &mut h265_caps as *mut _ as *mut c_void,
        flags: 0,
    };
    let mut caps = VkVideoCapabilitiesKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
        p_next: &mut decode_caps as *mut _ as *mut c_void,
        flags: 0,
        min_bitstream_buffer_offset_alignment: 0,
        min_bitstream_buffer_size_alignment: 0,
        picture_access_granularity: VkExtent2D::default(),
        min_coded_extent: VkExtent2D::default(),
        max_coded_extent: VkExtent2D::default(),
        max_dpb_slots: 0,
        max_active_reference_pictures: 0,
        std_header_version: VkExtensionProperties {
            extension_name: [0; VK_MAX_EXTENSION_NAME_SIZE],
            spec_version: 0,
        },
    };

    // SAFETY: every chain pointer references a local that lives until
    // the call returns; each struct's `sType` discriminant is set per
    // the spec.
    let result = unsafe { get_caps(physical_device.handle(), &profile, &mut caps) };
    if result != VK_SUCCESS {
        return Err(VkError::Result {
            op: "vkGetPhysicalDeviceVideoCapabilitiesKHR",
            result,
        });
    }

    Ok(VideoDecodeH265Capabilities {
        min_coded_extent: (caps.min_coded_extent.width, caps.min_coded_extent.height),
        max_coded_extent: (caps.max_coded_extent.width, caps.max_coded_extent.height),
        picture_access_granularity: (
            caps.picture_access_granularity.width,
            caps.picture_access_granularity.height,
        ),
        max_dpb_slots: caps.max_dpb_slots,
        max_active_reference_pictures: caps.max_active_reference_pictures,
        max_level_idc: h265_caps.max_level_idc,
        std_header_version: caps.std_header_version,
        capability_flags: caps.flags,
        decode_capability_flags: decode_caps.flags,
        min_bitstream_buffer_offset_alignment: caps.min_bitstream_buffer_offset_alignment,
        min_bitstream_buffer_size_alignment: caps.min_bitstream_buffer_size_alignment,
    })
}

// ─────────────────────────── AV1 capabilities ────────────────────────────────

/// Decoded form of the AV1 decode capability chain returned by
/// `vkGetPhysicalDeviceVideoCapabilitiesKHR`. Mirrors the H.264 /
/// H.265 shape with `max_level` from `VkVideoDecodeAV1CapabilitiesKHR`.
#[derive(Debug, Clone)]
pub struct VideoDecodeAV1Capabilities {
    /// Smallest image extent the decoder accepts.
    pub min_coded_extent: (u32, u32),
    /// Largest image extent the decoder will produce.
    pub max_coded_extent: (u32, u32),
    /// Picture-access granularity (super-block alignment for AV1).
    pub picture_access_granularity: (u32, u32),
    /// Maximum DPB slot count the implementation supports.
    pub max_dpb_slots: u32,
    /// Maximum count of active reference pictures simultaneously
    /// drawable from the DPB. AV1 uses 8 reference frames per the
    /// spec, but Vulkan may report fewer if the profile/level chosen
    /// is constrained.
    pub max_active_reference_pictures: u32,
    /// Maximum AV1 level the implementation supports for the queried
    /// profile, as a `StdVideoAV1Level` enum index.
    pub max_level: StdVideoAV1Level,
    /// Implementation-supplied `VkExtensionProperties` describing the
    /// `VK_STD_vulkan_video_codec_av1_decode` header version.
    pub std_header_version: VkExtensionProperties,
    /// Capability flags from `VkVideoCapabilitiesKHR.flags` (raw bits).
    pub capability_flags: u32,
    /// Decode-specific capability flags from
    /// `VkVideoDecodeCapabilitiesKHR.flags` (raw bits).
    pub decode_capability_flags: u32,
    /// Bitstream-buffer offset alignment.
    pub min_bitstream_buffer_offset_alignment: u64,
    /// Bitstream-buffer size alignment.
    pub min_bitstream_buffer_size_alignment: u64,
}

/// Run `vkGetPhysicalDeviceVideoCapabilitiesKHR` for an AV1 decode
/// profile.
///
/// `profile` is one of the `STD_VIDEO_AV1_PROFILE_*` constants in
/// [`crate::sys`]; in practice almost every dispatched query uses
/// `STD_VIDEO_AV1_PROFILE_MAIN` (the 8-bit / 10-bit 4:2:0 baseline).
///
/// `film_grain_support` should be `false` for the capability query —
/// the caller doesn't commit to providing film-grain params at this
/// stage; if `true` is needed the function will pass `1` to the
/// driver.
pub fn query_video_decode_av1_capabilities(
    physical_device: &PhysicalDevice<'_>,
    profile: StdVideoAV1Profile,
    film_grain_support: bool,
) -> Result<VideoDecodeAV1Capabilities, VkError> {
    let fns = physical_device.instance_fns();
    let get_caps =
        fns.get_physical_device_video_capabilities_khr
            .ok_or(VkError::MissingFunction(
                "vkGetPhysicalDeviceVideoCapabilitiesKHR",
            ))?;

    let av1_profile = VkVideoDecodeAV1ProfileInfoKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_PROFILE_INFO_KHR,
        p_next: ptr::null(),
        std_profile: profile,
        film_grain_support: u32::from(film_grain_support),
    };
    let profile_info = VkVideoProfileInfoKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
        p_next: &av1_profile as *const _ as *const c_void,
        video_codec_operation: VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR,
        chroma_subsampling: VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
        luma_bit_depth: VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
        chroma_bit_depth: VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
    };

    let mut av1_caps = VkVideoDecodeAV1CapabilitiesKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_CAPABILITIES_KHR,
        p_next: ptr::null_mut(),
        max_level: 0,
    };
    let mut decode_caps = VkVideoDecodeCapabilitiesKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_CAPABILITIES_KHR,
        p_next: &mut av1_caps as *mut _ as *mut c_void,
        flags: 0,
    };
    let mut caps = VkVideoCapabilitiesKHR {
        s_type: VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR,
        p_next: &mut decode_caps as *mut _ as *mut c_void,
        flags: 0,
        min_bitstream_buffer_offset_alignment: 0,
        min_bitstream_buffer_size_alignment: 0,
        picture_access_granularity: VkExtent2D::default(),
        min_coded_extent: VkExtent2D::default(),
        max_coded_extent: VkExtent2D::default(),
        max_dpb_slots: 0,
        max_active_reference_pictures: 0,
        std_header_version: VkExtensionProperties {
            extension_name: [0; VK_MAX_EXTENSION_NAME_SIZE],
            spec_version: 0,
        },
    };

    // SAFETY: chain pointers reference locals that live until the
    // call returns; each struct's `sType` is set per the spec.
    let result = unsafe { get_caps(physical_device.handle(), &profile_info, &mut caps) };
    if result != VK_SUCCESS {
        return Err(VkError::Result {
            op: "vkGetPhysicalDeviceVideoCapabilitiesKHR",
            result,
        });
    }

    Ok(VideoDecodeAV1Capabilities {
        min_coded_extent: (caps.min_coded_extent.width, caps.min_coded_extent.height),
        max_coded_extent: (caps.max_coded_extent.width, caps.max_coded_extent.height),
        picture_access_granularity: (
            caps.picture_access_granularity.width,
            caps.picture_access_granularity.height,
        ),
        max_dpb_slots: caps.max_dpb_slots,
        max_active_reference_pictures: caps.max_active_reference_pictures,
        max_level: av1_caps.max_level,
        std_header_version: caps.std_header_version,
        capability_flags: caps.flags,
        decode_capability_flags: decode_caps.flags,
        min_bitstream_buffer_offset_alignment: caps.min_bitstream_buffer_offset_alignment,
        min_bitstream_buffer_size_alignment: caps.min_bitstream_buffer_size_alignment,
    })
}

// ─────────────────────────── Video session ───────────────────────────────────

/// RAII wrapper over `VkVideoSessionKHR`.
///
/// `Drop` calls `vkDestroyVideoSessionKHR` and any
/// `VkDeviceMemory` allocations bound through
/// [`Self::allocate_and_bind_memory`] are released via
/// `vkFreeMemory`. The session is bound to the parent [`Device`]'s
/// lifetime by the borrow.
pub struct VideoSession<'device> {
    device: &'device Device,
    handle: VkVideoSessionKHR,
    /// Handles allocated and bound to this session. Freed in Drop.
    bound_memory: Vec<VkDeviceMemory>,
    /// Profile metadata, kept for debug.
    profile_idc: StdVideoH264ProfileIdc,
}

impl<'device> VideoSession<'device> {
    /// Construct a `VkVideoSessionKHR` for an H.264 decode profile.
    ///
    /// `caps` should be the result of
    /// [`query_video_decode_h264_capabilities`]; the function uses
    /// the std-header version and DPB-slot upper bounds reported
    /// there. `max_extent` is the largest decoded picture extent
    /// the session will be asked to handle (e.g. `(1920, 1088)` for
    /// HD H.264).
    ///
    /// `queue_family_index` must reference a video-decode queue
    /// family on the parent device.
    // matches the C `vkCreateVideoSessionKHR` + H.264 profile parameter set;
    // we keep the eight inputs to mirror the underlying FFI shape.
    #[allow(clippy::too_many_arguments)]
    pub fn new_h264_decode(
        device: &'device Device,
        physical_device: &PhysicalDevice<'_>,
        queue_family_index: u32,
        caps: &VideoDecodeH264Capabilities,
        max_extent: (u32, u32),
        profile_idc: StdVideoH264ProfileIdc,
        max_dpb_slots: u32,
        max_active_reference_pictures: u32,
    ) -> Result<Self, VkError> {
        // We take `physical_device` here for parity with the caps
        // call site (and so a future round can reach back to memory
        // properties without a second lookup), but the actual
        // session-create path doesn't need it; reference it here to
        // silence the unused-variable warning.
        let _ = physical_device;

        let h264_profile = VkVideoDecodeH264ProfileInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_PROFILE_INFO_KHR,
            p_next: ptr::null(),
            std_profile_idc: profile_idc,
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

        // Mirror the capabilities-reported header version verbatim
        // when present; otherwise advertise the spec
        // VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_SPEC_VERSION.
        let std_header_version = if caps.std_header_version.spec_version != 0 {
            caps.std_header_version
        } else {
            let mut name: [c_char; VK_MAX_EXTENSION_NAME_SIZE] = [0; VK_MAX_EXTENSION_NAME_SIZE];
            for (i, b) in VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_EXTENSION_NAME
                .as_bytes()
                .iter()
                .take(VK_MAX_EXTENSION_NAME_SIZE - 1)
                .enumerate()
            {
                name[i] = *b as c_char;
            }
            VkExtensionProperties {
                extension_name: name,
                spec_version: VK_STD_VULKAN_VIDEO_CODEC_H264_DECODE_SPEC_VERSION,
            }
        };

        let create_info = VkVideoSessionCreateInfoKHR {
            s_type: VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
            p_next: ptr::null(),
            queue_family_index,
            flags: 0,
            p_video_profile: &profile,
            picture_format: VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
            max_coded_extent: VkExtent2D {
                width: max_extent.0,
                height: max_extent.1,
            },
            reference_picture_format: VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
            max_dpb_slots: max_dpb_slots.min(caps.max_dpb_slots.max(1)),
            max_active_reference_pictures: max_active_reference_pictures
                .min(caps.max_active_reference_pictures.max(1)),
            p_std_header_version: &std_header_version,
        };

        let mut handle: VkVideoSessionKHR = ptr::null_mut();
        // SAFETY: every pNext chain pointer references a local that
        // lives until the call returns. Vulkan must internalise any
        // referenced data before returning.
        let result = unsafe {
            (device.fns().create_video_session_khr)(
                device.handle(),
                &create_info,
                ptr::null(),
                &mut handle,
            )
        };
        if result != VK_SUCCESS {
            return Err(VkError::Result {
                op: "vkCreateVideoSessionKHR",
                result,
            });
        }

        Ok(Self {
            device,
            handle,
            bound_memory: Vec::new(),
            profile_idc,
        })
    }

    /// Raw `VkVideoSessionKHR` handle.
    pub fn handle(&self) -> VkVideoSessionKHR {
        self.handle
    }

    /// `profile_idc` the session was created against.
    pub fn profile_idc(&self) -> StdVideoH264ProfileIdc {
        self.profile_idc
    }

    /// Take ownership of the session handle + bound memory list,
    /// neutering this `VideoSession` so its `Drop` becomes a no-op.
    /// The caller assumes responsibility for calling
    /// `vkDestroyVideoSessionKHR` and `vkFreeMemory` for each
    /// returned handle. Used by `DecoderState::drop` to avoid
    /// dispatching through a possibly-dangling `&Device` borrow when
    /// the parent struct's field-order tear-down is in progress.
    pub fn detach(&mut self) -> (VkVideoSessionKHR, Vec<VkDeviceMemory>) {
        let handle = std::mem::replace(&mut self.handle, ptr::null_mut());
        let memory = std::mem::take(&mut self.bound_memory);
        (handle, memory)
    }

    /// Query memory requirements for this session via
    /// `vkGetVideoSessionMemoryRequirementsKHR`.
    ///
    /// The two-call enumerate pattern: count probe followed by sized
    /// fetch. Each entry gives the alignment, size, and acceptable
    /// memory-type bitmask for a single memory bind index that the
    /// caller must satisfy with `vkBindVideoSessionMemoryKHR`.
    pub fn memory_requirements(&self) -> Result<Vec<VkVideoSessionMemoryRequirementsKHR>, VkError> {
        let mut count: u32 = 0;
        // SAFETY: count probe with null sized buffer.
        let result = unsafe {
            (self.device.fns().get_video_session_memory_requirements_khr)(
                self.device.handle(),
                self.handle,
                &mut count,
                ptr::null_mut(),
            )
        };
        if result != VK_SUCCESS {
            return Err(VkError::Result {
                op: "vkGetVideoSessionMemoryRequirementsKHR",
                result,
            });
        }

        if count == 0 {
            return Ok(Vec::new());
        }

        let mut reqs: Vec<VkVideoSessionMemoryRequirementsKHR> = (0..count)
            .map(|_| VkVideoSessionMemoryRequirementsKHR {
                s_type: VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR,
                p_next: ptr::null_mut(),
                memory_bind_index: 0,
                memory_requirements: VkMemoryRequirements::default(),
            })
            .collect();

        // SAFETY: `reqs` is sized from the probe; Vulkan writes at
        // most `count` entries.
        let result = unsafe {
            (self.device.fns().get_video_session_memory_requirements_khr)(
                self.device.handle(),
                self.handle,
                &mut count,
                reqs.as_mut_ptr(),
            )
        };
        if result != VK_SUCCESS {
            return Err(VkError::Result {
                op: "vkGetVideoSessionMemoryRequirementsKHR",
                result,
            });
        }
        reqs.truncate(count as usize);
        Ok(reqs)
    }

    /// Allocate and bind device memory for every requirement
    /// reported by [`Self::memory_requirements`].
    ///
    /// For each requirement we pick the first memory type from the
    /// physical device's memory-properties table whose bit is set in
    /// `memory_type_bits` and whose `propertyFlags` include
    /// `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`. This is the universal
    /// fallback for video session backing memory; spec doesn't
    /// require any host-visible bit.
    ///
    /// On success the session retains ownership of the allocations
    /// (freed on `Drop`). On failure any allocations made before the
    /// failure are released and the original memory state of the
    /// session is preserved.
    pub fn allocate_and_bind_memory(
        &mut self,
        physical_device: &PhysicalDevice<'_>,
    ) -> Result<usize, VkError> {
        let reqs = self.memory_requirements()?;
        if reqs.is_empty() {
            return Ok(0);
        }

        let mem_props = physical_device_memory_properties(physical_device);

        let mut allocations: Vec<VkDeviceMemory> = Vec::with_capacity(reqs.len());
        let mut binds: Vec<VkBindVideoSessionMemoryInfoKHR> = Vec::with_capacity(reqs.len());

        for req in &reqs {
            let type_index = pick_memory_type(
                &mem_props,
                req.memory_requirements.memory_type_bits,
                sys::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            )
            .or_else(|| {
                // Some implementations report 0 for the device-local
                // bit on sysram-only systems; fall back to "any
                // satisfying type" if no device-local match.
                pick_memory_type(&mem_props, req.memory_requirements.memory_type_bits, 0)
            })
            .ok_or(VkError::MissingFunction(
                "no suitable memory type for video session bind",
            ))?;

            let alloc_info = VkMemoryAllocateInfo {
                s_type: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                p_next: ptr::null(),
                allocation_size: req.memory_requirements.size,
                memory_type_index: type_index,
            };
            let mut mem: VkDeviceMemory = ptr::null_mut();
            // SAFETY: alloc_info is fully populated; passing null
            // allocator selects the implementation default.
            let result = unsafe {
                (self.device.fns().allocate_memory)(
                    self.device.handle(),
                    &alloc_info,
                    ptr::null(),
                    &mut mem,
                )
            };
            if result != VK_SUCCESS {
                // Roll back any earlier allocations so we don't leak
                // when the binding fails partway through.
                for m in &allocations {
                    // SAFETY: each `m` was returned by `vkAllocateMemory`
                    // and has not been freed yet.
                    unsafe {
                        (self.device.fns().free_memory)(self.device.handle(), *m, ptr::null());
                    }
                }
                return Err(VkError::Result {
                    op: "vkAllocateMemory",
                    result,
                });
            }
            allocations.push(mem);
            binds.push(VkBindVideoSessionMemoryInfoKHR {
                s_type: VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR,
                p_next: ptr::null(),
                memory_bind_index: req.memory_bind_index,
                memory: mem,
                memory_offset: 0,
                memory_size: req.memory_requirements.size,
            });
        }

        // SAFETY: every memory handle in `binds` was just allocated;
        // every `memory_bind_index` came from the requirements
        // emitted by the same session.
        let result = unsafe {
            (self.device.fns().bind_video_session_memory_khr)(
                self.device.handle(),
                self.handle,
                binds.len() as u32,
                binds.as_ptr(),
            )
        };
        if result != VK_SUCCESS {
            for m in &allocations {
                // SAFETY: each `m` was returned by vkAllocateMemory.
                unsafe {
                    (self.device.fns().free_memory)(self.device.handle(), *m, ptr::null());
                }
            }
            return Err(VkError::Result {
                op: "vkBindVideoSessionMemoryKHR",
                result,
            });
        }

        let count = allocations.len();
        self.bound_memory.extend(allocations);
        Ok(count)
    }
}

impl<'device> Drop for VideoSession<'device> {
    fn drop(&mut self) {
        // Destroy the session first; per the spec, freeing the
        // underlying memory before destroying the session is
        // undefined.
        if !self.handle.is_null() {
            // SAFETY: handle was created by vkCreateVideoSessionKHR
            // and has not been previously destroyed.
            unsafe {
                (self.device.fns().destroy_video_session_khr)(
                    self.device.handle(),
                    self.handle,
                    ptr::null(),
                );
            }
            self.handle = ptr::null_mut();
        }
        // Now release the backing allocations.
        for m in self.bound_memory.drain(..) {
            // SAFETY: `m` was returned by vkAllocateMemory above.
            unsafe {
                (self.device.fns().free_memory)(self.device.handle(), m, ptr::null());
            }
        }
    }
}

// ─────────────────────────── helpers ─────────────────────────────────────────

/// Read the `VkPhysicalDeviceMemoryProperties` table for a physical
/// device. Wrapped here so the unsafe-block surface in
/// [`VideoSession::allocate_and_bind_memory`] stays small.
fn physical_device_memory_properties(
    physical_device: &PhysicalDevice<'_>,
) -> VkPhysicalDeviceMemoryProperties {
    // SAFETY: zero-initialised storage of the right size+layout for
    // VkPhysicalDeviceMemoryProperties; Vulkan writes every
    // populated field.
    let mut props: VkPhysicalDeviceMemoryProperties = unsafe { std::mem::zeroed() };
    // SAFETY: pointer into a freshly zeroed local buffer of the
    // right type; `physical_device` handle is owned by the parent
    // instance and has not been destroyed.
    unsafe {
        (physical_device
            .instance_fns()
            .get_physical_device_memory_properties)(physical_device.handle(), &mut props);
    }
    props
}

/// Pick the first memory type from `props` whose bit is set in
/// `type_bits` AND whose `propertyFlags` contains every bit in
/// `required_flags`. Returns `None` if no entry matches (caller
/// typically retries with `required_flags = 0`).
fn pick_memory_type(
    props: &VkPhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: u32,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if type_bits & (1u32 << i) == 0 {
            continue;
        }
        let mt = &props.memory_types[i as usize];
        if (mt.property_flags & required_flags) == required_flags {
            return Some(i);
        }
    }
    None
}
