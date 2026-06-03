//! Verify Rust struct sizes match the C ABI from `vulkan_video_codec_h264std.h`.
#![cfg(any(target_os = "linux", target_os = "windows"))]

use oxideav_vulkan_video::sys::*;

#[test]
fn h264_std_struct_sizes_match_c_abi() {
    assert_eq!(std::mem::size_of::<StdVideoH264SpsFlags>(), 4);
    assert_eq!(std::mem::size_of::<StdVideoH264PpsFlags>(), 4);
    assert_eq!(
        std::mem::size_of::<StdVideoH264SequenceParameterSet>(),
        88,
        "StdVideoH264SequenceParameterSet size mismatch"
    );
    assert_eq!(
        std::mem::size_of::<StdVideoH264PictureParameterSet>(),
        24,
        "StdVideoH264PictureParameterSet size mismatch"
    );
    assert_eq!(std::mem::size_of::<StdVideoDecodeH264PictureInfo>(), 20);
    assert_eq!(std::mem::size_of::<StdVideoDecodeH264ReferenceInfo>(), 16);
    assert_eq!(
        std::mem::size_of::<VkVideoDecodeH264SessionParametersCreateInfoKHR>(),
        32
    );
    assert_eq!(
        std::mem::size_of::<VkVideoDecodeH264SessionParametersAddInfoKHR>(),
        48
    );
    assert_eq!(
        std::mem::size_of::<VkVideoSessionParametersCreateInfoKHR>(),
        40
    );
}

#[test]
fn vulkan_round4_struct_sizes_match_c_abi() {
    assert_eq!(std::mem::size_of::<VkVideoDecodeInfoKHR>(), 120);
    assert_eq!(std::mem::size_of::<VkVideoBeginCodingInfoKHR>(), 56);
    assert_eq!(std::mem::size_of::<VkVideoEndCodingInfoKHR>(), 24);
    assert_eq!(std::mem::size_of::<VkVideoCodingControlInfoKHR>(), 24);
    assert_eq!(std::mem::size_of::<VkVideoPictureResourceInfoKHR>(), 48);
    assert_eq!(std::mem::size_of::<VkVideoReferenceSlotInfoKHR>(), 32);
    assert_eq!(std::mem::size_of::<VkVideoDecodeH264PictureInfoKHR>(), 40);
    assert_eq!(std::mem::size_of::<VkVideoDecodeH264DpbSlotInfoKHR>(), 24);
    assert_eq!(std::mem::size_of::<VkVideoProfileListInfoKHR>(), 32);
    assert_eq!(std::mem::size_of::<VkBufferCreateInfo>(), 56);
    assert_eq!(std::mem::size_of::<VkImageCreateInfo>(), 88);
    assert_eq!(std::mem::size_of::<VkImageViewCreateInfo>(), 80);
    assert_eq!(std::mem::size_of::<VkImageMemoryBarrier>(), 72);
    assert_eq!(std::mem::size_of::<VkSubmitInfo>(), 72);
    assert_eq!(std::mem::size_of::<VkBufferImageCopy>(), 56);
}

/// Per-field byte offsets of every Vulkan / std video struct used by
/// the Round 4 decode pipeline. Cross-checked against `cc -I/usr/include`
/// of `<vulkan/vulkan_core.h>` + `<vk_video/vulkan_video_codec_h264std*.h>`
/// using a one-shot `offsetof` walker; these are the values the C ABI
/// reports on x86_64 Linux against the spec-published Khronos headers.
///
/// A mismatch here means the FFI struct is layout-broken even if the
/// total `sizeof` matches, which is how a Round 4-style "the driver
/// crashes deep inside its own dispatch" bug would show up if we got
/// a field shifted by 4 bytes (e.g. forgetting a reserved padding
/// member).
#[test]
fn h264_std_struct_field_offsets_match_c_abi() {
    use std::mem::offset_of;

    // ── StdVideoH264SequenceParameterSet ─────────────────────
    assert_eq!(offset_of!(StdVideoH264SequenceParameterSet, flags), 0);
    assert_eq!(offset_of!(StdVideoH264SequenceParameterSet, profile_idc), 4);
    assert_eq!(offset_of!(StdVideoH264SequenceParameterSet, level_idc), 8);
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, chroma_format_idc),
        12
    );
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, seq_parameter_set_id),
        16
    );
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, bit_depth_luma_minus8),
        17
    );
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, bit_depth_chroma_minus8),
        18
    );
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, log2_max_frame_num_minus4),
        19
    );
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, pic_order_cnt_type),
        20
    );
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, offset_for_non_ref_pic),
        24
    );
    assert_eq!(
        offset_of!(
            StdVideoH264SequenceParameterSet,
            offset_for_top_to_bottom_field
        ),
        28
    );
    assert_eq!(
        offset_of!(
            StdVideoH264SequenceParameterSet,
            log2_max_pic_order_cnt_lsb_minus4
        ),
        32
    );
    assert_eq!(
        offset_of!(
            StdVideoH264SequenceParameterSet,
            num_ref_frames_in_pic_order_cnt_cycle
        ),
        33
    );
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, max_num_ref_frames),
        34
    );
    assert_eq!(offset_of!(StdVideoH264SequenceParameterSet, reserved1), 35);
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, pic_width_in_mbs_minus1),
        36
    );
    assert_eq!(
        offset_of!(
            StdVideoH264SequenceParameterSet,
            pic_height_in_map_units_minus1
        ),
        40
    );
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, frame_crop_left_offset),
        44
    );
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, frame_crop_right_offset),
        48
    );
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, frame_crop_top_offset),
        52
    );
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, frame_crop_bottom_offset),
        56
    );
    assert_eq!(offset_of!(StdVideoH264SequenceParameterSet, reserved2), 60);
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, p_offset_for_ref_frame),
        64
    );
    assert_eq!(
        offset_of!(StdVideoH264SequenceParameterSet, p_scaling_lists),
        72
    );
    assert_eq!(
        offset_of!(
            StdVideoH264SequenceParameterSet,
            p_sequence_parameter_set_vui
        ),
        80
    );

    // ── StdVideoH264PictureParameterSet ──────────────────────
    assert_eq!(offset_of!(StdVideoH264PictureParameterSet, flags), 0);
    assert_eq!(
        offset_of!(StdVideoH264PictureParameterSet, seq_parameter_set_id),
        4
    );
    assert_eq!(
        offset_of!(StdVideoH264PictureParameterSet, pic_parameter_set_id),
        5
    );
    assert_eq!(
        offset_of!(
            StdVideoH264PictureParameterSet,
            num_ref_idx_l0_default_active_minus1
        ),
        6
    );
    assert_eq!(
        offset_of!(
            StdVideoH264PictureParameterSet,
            num_ref_idx_l1_default_active_minus1
        ),
        7
    );
    assert_eq!(
        offset_of!(StdVideoH264PictureParameterSet, weighted_bipred_idc),
        8
    );
    assert_eq!(
        offset_of!(StdVideoH264PictureParameterSet, pic_init_qp_minus26),
        12
    );
    assert_eq!(
        offset_of!(StdVideoH264PictureParameterSet, pic_init_qs_minus26),
        13
    );
    assert_eq!(
        offset_of!(StdVideoH264PictureParameterSet, chroma_qp_index_offset),
        14
    );
    assert_eq!(
        offset_of!(
            StdVideoH264PictureParameterSet,
            second_chroma_qp_index_offset
        ),
        15
    );
    assert_eq!(
        offset_of!(StdVideoH264PictureParameterSet, p_scaling_lists),
        16
    );

    // ── StdVideoDecodeH264PictureInfo ────────────────────────
    assert_eq!(offset_of!(StdVideoDecodeH264PictureInfo, flags), 0);
    assert_eq!(
        offset_of!(StdVideoDecodeH264PictureInfo, seq_parameter_set_id),
        4
    );
    assert_eq!(
        offset_of!(StdVideoDecodeH264PictureInfo, pic_parameter_set_id),
        5
    );
    assert_eq!(offset_of!(StdVideoDecodeH264PictureInfo, reserved1), 6);
    assert_eq!(offset_of!(StdVideoDecodeH264PictureInfo, reserved2), 7);
    assert_eq!(offset_of!(StdVideoDecodeH264PictureInfo, frame_num), 8);
    assert_eq!(offset_of!(StdVideoDecodeH264PictureInfo, idr_pic_id), 10);
    assert_eq!(offset_of!(StdVideoDecodeH264PictureInfo, pic_order_cnt), 12);

    // ── StdVideoDecodeH264ReferenceInfo ──────────────────────
    assert_eq!(offset_of!(StdVideoDecodeH264ReferenceInfo, flags), 0);
    assert_eq!(offset_of!(StdVideoDecodeH264ReferenceInfo, frame_num), 4);
    assert_eq!(offset_of!(StdVideoDecodeH264ReferenceInfo, reserved), 6);
    assert_eq!(
        offset_of!(StdVideoDecodeH264ReferenceInfo, pic_order_cnt),
        8
    );
}

#[test]
fn vulkan_round4_struct_field_offsets_match_c_abi() {
    use std::mem::offset_of;

    // ── VkVideoDecodeInfoKHR ─────────────────────────────────
    assert_eq!(offset_of!(VkVideoDecodeInfoKHR, s_type), 0);
    assert_eq!(offset_of!(VkVideoDecodeInfoKHR, p_next), 8);
    assert_eq!(offset_of!(VkVideoDecodeInfoKHR, flags), 16);
    assert_eq!(offset_of!(VkVideoDecodeInfoKHR, src_buffer), 24);
    assert_eq!(offset_of!(VkVideoDecodeInfoKHR, src_buffer_offset), 32);
    assert_eq!(offset_of!(VkVideoDecodeInfoKHR, src_buffer_range), 40);
    assert_eq!(offset_of!(VkVideoDecodeInfoKHR, dst_picture_resource), 48);
    assert_eq!(offset_of!(VkVideoDecodeInfoKHR, p_setup_reference_slot), 96);
    assert_eq!(offset_of!(VkVideoDecodeInfoKHR, reference_slot_count), 104);
    assert_eq!(offset_of!(VkVideoDecodeInfoKHR, p_reference_slots), 112);

    // ── VkVideoBeginCodingInfoKHR ────────────────────────────
    assert_eq!(offset_of!(VkVideoBeginCodingInfoKHR, s_type), 0);
    assert_eq!(offset_of!(VkVideoBeginCodingInfoKHR, p_next), 8);
    assert_eq!(offset_of!(VkVideoBeginCodingInfoKHR, flags), 16);
    assert_eq!(offset_of!(VkVideoBeginCodingInfoKHR, video_session), 24);
    assert_eq!(
        offset_of!(VkVideoBeginCodingInfoKHR, video_session_parameters),
        32
    );
    assert_eq!(
        offset_of!(VkVideoBeginCodingInfoKHR, reference_slot_count),
        40
    );
    assert_eq!(offset_of!(VkVideoBeginCodingInfoKHR, p_reference_slots), 48);

    // ── VkVideoReferenceSlotInfoKHR ──────────────────────────
    assert_eq!(offset_of!(VkVideoReferenceSlotInfoKHR, s_type), 0);
    assert_eq!(offset_of!(VkVideoReferenceSlotInfoKHR, p_next), 8);
    assert_eq!(offset_of!(VkVideoReferenceSlotInfoKHR, slot_index), 16);
    assert_eq!(
        offset_of!(VkVideoReferenceSlotInfoKHR, p_picture_resource),
        24
    );

    // ── VkVideoPictureResourceInfoKHR ────────────────────────
    assert_eq!(offset_of!(VkVideoPictureResourceInfoKHR, s_type), 0);
    assert_eq!(offset_of!(VkVideoPictureResourceInfoKHR, p_next), 8);
    assert_eq!(offset_of!(VkVideoPictureResourceInfoKHR, coded_offset), 16);
    assert_eq!(offset_of!(VkVideoPictureResourceInfoKHR, coded_extent), 24);
    assert_eq!(
        offset_of!(VkVideoPictureResourceInfoKHR, base_array_layer),
        32
    );
    assert_eq!(
        offset_of!(VkVideoPictureResourceInfoKHR, image_view_binding),
        40
    );

    // ── VkVideoDecodeH264PictureInfoKHR ──────────────────────
    assert_eq!(offset_of!(VkVideoDecodeH264PictureInfoKHR, s_type), 0);
    assert_eq!(offset_of!(VkVideoDecodeH264PictureInfoKHR, p_next), 8);
    assert_eq!(
        offset_of!(VkVideoDecodeH264PictureInfoKHR, p_std_picture_info),
        16
    );
    assert_eq!(offset_of!(VkVideoDecodeH264PictureInfoKHR, slice_count), 24);
    assert_eq!(
        offset_of!(VkVideoDecodeH264PictureInfoKHR, p_slice_offsets),
        32
    );

    // ── VkVideoDecodeH264DpbSlotInfoKHR ──────────────────────
    assert_eq!(offset_of!(VkVideoDecodeH264DpbSlotInfoKHR, s_type), 0);
    assert_eq!(offset_of!(VkVideoDecodeH264DpbSlotInfoKHR, p_next), 8);
    assert_eq!(
        offset_of!(VkVideoDecodeH264DpbSlotInfoKHR, p_std_reference_info),
        16
    );

    // ── VkVideoDecodeH264SessionParametersAddInfoKHR ─────────
    assert_eq!(
        offset_of!(VkVideoDecodeH264SessionParametersAddInfoKHR, s_type),
        0
    );
    assert_eq!(
        offset_of!(VkVideoDecodeH264SessionParametersAddInfoKHR, p_next),
        8
    );
    assert_eq!(
        offset_of!(VkVideoDecodeH264SessionParametersAddInfoKHR, std_sps_count),
        16
    );
    assert_eq!(
        offset_of!(VkVideoDecodeH264SessionParametersAddInfoKHR, p_std_sp_ss),
        24
    );
    assert_eq!(
        offset_of!(VkVideoDecodeH264SessionParametersAddInfoKHR, std_pps_count),
        32
    );
    assert_eq!(
        offset_of!(VkVideoDecodeH264SessionParametersAddInfoKHR, p_std_pp_ss),
        40
    );

    // ── VkVideoSessionParametersCreateInfoKHR ────────────────
    assert_eq!(offset_of!(VkVideoSessionParametersCreateInfoKHR, s_type), 0);
    assert_eq!(offset_of!(VkVideoSessionParametersCreateInfoKHR, p_next), 8);
    assert_eq!(offset_of!(VkVideoSessionParametersCreateInfoKHR, flags), 16);
    assert_eq!(
        offset_of!(
            VkVideoSessionParametersCreateInfoKHR,
            video_session_parameters_template
        ),
        24
    );
    assert_eq!(
        offset_of!(VkVideoSessionParametersCreateInfoKHR, video_session),
        32
    );

    // ── VkVideoProfileListInfoKHR ────────────────────────────
    assert_eq!(offset_of!(VkVideoProfileListInfoKHR, s_type), 0);
    assert_eq!(offset_of!(VkVideoProfileListInfoKHR, p_next), 8);
    assert_eq!(offset_of!(VkVideoProfileListInfoKHR, profile_count), 16);
    assert_eq!(offset_of!(VkVideoProfileListInfoKHR, p_profiles), 24);
}

/// Round 8: H.265 and AV1 decode profile/capability structs added to
/// `sys.rs` so `engine_info()` can populate the matching
/// `HwCodecCaps` rows with real max-extent / level / DPB-slot
/// numbers. Sizes / offsets verified against `cc -I…` of
/// `<vulkan/vulkan_core.h>` on x86_64 Linux against the spec-published
/// Khronos headers (`VkVideoDecodeH265ProfileInfoKHR`,
/// `VkVideoDecodeH265CapabilitiesKHR`, `VkVideoDecodeAV1ProfileInfoKHR`,
/// `VkVideoDecodeAV1CapabilitiesKHR` — each 24 bytes with the field
/// offsets below).
#[test]
fn vulkan_round8_h265_av1_caps_struct_sizes_match_c_abi() {
    assert_eq!(std::mem::size_of::<VkVideoDecodeH265ProfileInfoKHR>(), 24);
    assert_eq!(std::mem::size_of::<VkVideoDecodeH265CapabilitiesKHR>(), 24);
    assert_eq!(std::mem::size_of::<VkVideoDecodeAV1ProfileInfoKHR>(), 24);
    assert_eq!(std::mem::size_of::<VkVideoDecodeAV1CapabilitiesKHR>(), 24);
}

#[test]
fn vulkan_round8_h265_av1_caps_struct_field_offsets_match_c_abi() {
    use std::mem::offset_of;

    // ── VkVideoDecodeH265ProfileInfoKHR ──────────────────────
    assert_eq!(offset_of!(VkVideoDecodeH265ProfileInfoKHR, s_type), 0);
    assert_eq!(offset_of!(VkVideoDecodeH265ProfileInfoKHR, p_next), 8);
    assert_eq!(
        offset_of!(VkVideoDecodeH265ProfileInfoKHR, std_profile_idc),
        16
    );

    // ── VkVideoDecodeH265CapabilitiesKHR ─────────────────────
    assert_eq!(offset_of!(VkVideoDecodeH265CapabilitiesKHR, s_type), 0);
    assert_eq!(offset_of!(VkVideoDecodeH265CapabilitiesKHR, p_next), 8);
    assert_eq!(
        offset_of!(VkVideoDecodeH265CapabilitiesKHR, max_level_idc),
        16
    );

    // ── VkVideoDecodeAV1ProfileInfoKHR ───────────────────────
    assert_eq!(offset_of!(VkVideoDecodeAV1ProfileInfoKHR, s_type), 0);
    assert_eq!(offset_of!(VkVideoDecodeAV1ProfileInfoKHR, p_next), 8);
    assert_eq!(offset_of!(VkVideoDecodeAV1ProfileInfoKHR, std_profile), 16);
    assert_eq!(
        offset_of!(VkVideoDecodeAV1ProfileInfoKHR, film_grain_support),
        20
    );

    // ── VkVideoDecodeAV1CapabilitiesKHR ──────────────────────
    assert_eq!(offset_of!(VkVideoDecodeAV1CapabilitiesKHR, s_type), 0);
    assert_eq!(offset_of!(VkVideoDecodeAV1CapabilitiesKHR, p_next), 8);
    assert_eq!(offset_of!(VkVideoDecodeAV1CapabilitiesKHR, max_level), 16);
}
