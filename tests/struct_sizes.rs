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
