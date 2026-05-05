//! Minimal H.264 Annex-B / RBSP parser.
//!
//! Round 4 only needs enough parsing to:
//!
//! 1. Walk an Annex-B stream and yield NAL units (start-code stripped),
//!    classifying each by NAL type (SPS = 7, PPS = 8, IDR slice = 5,
//!    non-IDR slice = 1, …).
//! 2. Decode an SPS NAL into the subset of
//!    `StdVideoH264SequenceParameterSet` fields that the GPU will look
//!    at: profile_idc, level_idc, chroma_format_idc, log2_max_frame_num
//!    minus4, pic_order_cnt_type, max_num_ref_frames,
//!    pic_width_in_mbs_minus1, pic_height_in_map_units_minus1,
//!    frame_mbs_only_flag, direct_8x8_inference_flag.
//! 3. Decode a PPS NAL into the corresponding subset of
//!    `StdVideoH264PictureParameterSet`: pic_parameter_set_id,
//!    seq_parameter_set_id, entropy_coding_mode_flag,
//!    bottom_field_pic_order_in_frame_present_flag,
//!    num_ref_idx_l0/l1_default_active_minus1, weighted_pred_flag,
//!    weighted_bipred_idc, pic_init_qp_minus26, pic_init_qs_minus26,
//!    chroma_qp_index_offset, deblocking_filter_control_present_flag,
//!    constrained_intra_pred_flag, redundant_pic_cnt_present_flag.
//!
//! VUI / HRD / scaling-list parsing is intentionally NOT implemented —
//! the decoder's `pSequenceParameterSetVui` / `pScalingLists` are left
//! null and the relevant flags cleared. For an IDR-only High-profile
//! stream (the Round 4 test fixture) that is sufficient.

/// NAL unit type masks (`nal_unit_type` is the low 5 bits of the
/// first byte of an H.264 NAL).
pub mod nal_type {
    pub const SLICE: u8 = 1;
    pub const SLICE_IDR: u8 = 5;
    pub const SEI: u8 = 6;
    pub const SPS: u8 = 7;
    pub const PPS: u8 = 8;
}

/// Iterate the Annex-B start-code-prefixed NAL units in `buf`.
///
/// Returns each NAL unit as a slice into `buf` *without* the start
/// code. Trailing zero bytes that belong to the next start code's
/// `0x000001` prefix are stripped.
pub fn iter_nals(buf: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    let len = buf.len();

    loop {
        // Find next start code (000001 or 00000001) starting at pos.
        let mut sc = None;
        let mut p = pos;
        while p + 3 <= len {
            if buf[p] == 0 && buf[p + 1] == 0 {
                if buf[p + 2] == 1 {
                    sc = Some((p, 3));
                    break;
                }
                if p + 4 <= len && buf[p + 2] == 0 && buf[p + 3] == 1 {
                    sc = Some((p, 4));
                    break;
                }
            }
            p += 1;
        }
        let (sc_pos, sc_len) = match sc {
            Some(s) => s,
            None => break,
        };
        let nal_start = sc_pos + sc_len;
        // Find end (next start code or eof).
        let mut q = nal_start;
        let mut nal_end = len;
        while q + 3 <= len {
            if buf[q] == 0 && buf[q + 1] == 0 {
                if buf[q + 2] == 1 || (q + 4 <= len && buf[q + 2] == 0 && buf[q + 3] == 1) {
                    nal_end = q;
                    // Strip trailing zeros before this start code.
                    while nal_end > nal_start && buf[nal_end - 1] == 0 {
                        nal_end -= 1;
                    }
                    break;
                }
            }
            q += 1;
        }
        if nal_end > nal_start {
            out.push(&buf[nal_start..nal_end]);
        }
        pos = nal_end;
    }
    out
}

/// Parse an H.264 Annex-B / RBSP byte stream by stripping
/// emulation-prevention bytes (a 0x03 inserted between any two
/// successive 0x00 0x00 in the NAL payload).
pub fn rbsp_strip(nal: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(nal.len());
    let mut i = 0;
    while i < nal.len() {
        if i + 2 < nal.len() && nal[i] == 0 && nal[i + 1] == 0 && nal[i + 2] == 0x03 {
            out.push(0);
            out.push(0);
            i += 3;
        } else {
            out.push(nal[i]);
            i += 1;
        }
    }
    out
}

/// MSB-first variable-length bitstream reader for H.264 RBSP.
pub struct BitReader<'a> {
    buf: &'a [u8],
    /// Current bit offset (0..8 * buf.len()).
    bit_pos: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, bit_pos: 0 }
    }

    /// Read the next `n` bits (0 ≤ n ≤ 32) as a u32, MSB first.
    pub fn read_bits(&mut self, n: u32) -> u32 {
        let mut v: u32 = 0;
        for _ in 0..n {
            v = (v << 1) | self.read_bit() as u32;
        }
        v
    }

    /// Read one bit. Returns 0 if past end (clamps).
    pub fn read_bit(&mut self) -> u8 {
        let byte = self.bit_pos >> 3;
        if byte >= self.buf.len() {
            return 0;
        }
        let off = 7 - (self.bit_pos & 7);
        self.bit_pos += 1;
        (self.buf[byte] >> off) & 1
    }

    /// Read an unsigned Exp-Golomb code (`ue(v)`).
    pub fn read_ue(&mut self) -> u32 {
        let mut zeros = 0u32;
        while self.read_bit() == 0 {
            zeros += 1;
            if zeros > 32 {
                return 0;
            }
        }
        if zeros == 0 {
            return 0;
        }
        let suffix = self.read_bits(zeros);
        (1u32 << zeros) - 1 + suffix
    }

    /// Read a signed Exp-Golomb code (`se(v)`).
    pub fn read_se(&mut self) -> i32 {
        let v = self.read_ue();
        if v & 1 == 1 {
            ((v + 1) >> 1) as i32
        } else {
            -((v >> 1) as i32)
        }
    }
}

/// Decoded subset of `StdVideoH264SequenceParameterSet`.
#[derive(Debug, Clone, Default)]
pub struct ParsedSps {
    pub seq_parameter_set_id: u8,
    pub profile_idc: u8,
    pub level_idc: u8,
    pub chroma_format_idc: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub log2_max_frame_num_minus4: u8,
    pub pic_order_cnt_type: u8,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub max_num_ref_frames: u8,
    pub gaps_in_frame_num_value_allowed_flag: u8,
    pub pic_width_in_mbs_minus1: u32,
    pub pic_height_in_map_units_minus1: u32,
    pub frame_mbs_only_flag: u8,
    pub mb_adaptive_frame_field_flag: u8,
    pub direct_8x8_inference_flag: u8,
    pub frame_cropping_flag: u8,
    pub frame_crop_left_offset: u32,
    pub frame_crop_right_offset: u32,
    pub frame_crop_top_offset: u32,
    pub frame_crop_bottom_offset: u32,
    pub vui_parameters_present_flag: u8,
    pub constraint_set_flags: u8,
    /// Decoded image width in luma samples (post-crop).
    pub coded_width: u32,
    /// Decoded image height in luma samples (post-crop).
    pub coded_height: u32,
    /// Pre-crop width (mbs * 16).
    pub raw_width: u32,
    /// Pre-crop height (mbs * 16 / (interlaced ? 2 : 1)).
    pub raw_height: u32,
}

/// Parse an SPS NAL (NAL-byte included). Returns `None` on malformed
/// input or an unsupported profile.
pub fn parse_sps(nal: &[u8]) -> Option<ParsedSps> {
    if nal.is_empty() || (nal[0] & 0x1F) != nal_type::SPS {
        return None;
    }
    let rbsp = rbsp_strip(&nal[1..]);
    let mut r = BitReader::new(&rbsp);

    let mut s = ParsedSps::default();
    s.profile_idc = r.read_bits(8) as u8;
    s.constraint_set_flags = r.read_bits(8) as u8;
    s.level_idc = r.read_bits(8) as u8;
    s.seq_parameter_set_id = r.read_ue() as u8;

    // For the High family (and a few others) chroma_format_idc and the
    // bit-depth fields are present.
    let high_family = matches!(
        s.profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    );
    if high_family {
        s.chroma_format_idc = r.read_ue() as u8;
        if s.chroma_format_idc == 3 {
            // separate_colour_plane_flag
            r.read_bit();
        }
        s.bit_depth_luma_minus8 = r.read_ue() as u8;
        s.bit_depth_chroma_minus8 = r.read_ue() as u8;
        // qpprime_y_zero_transform_bypass_flag
        r.read_bit();
        // seq_scaling_matrix_present_flag
        let sl = r.read_bit();
        if sl == 1 {
            let n = if s.chroma_format_idc == 3 { 12 } else { 8 };
            for i in 0..n {
                let present = r.read_bit();
                if present == 1 {
                    let size = if i < 6 { 16 } else { 64 };
                    let mut last_scale: i32 = 8;
                    let mut next_scale: i32 = 8;
                    for _ in 0..size {
                        if next_scale != 0 {
                            let delta = r.read_se();
                            next_scale = (last_scale + delta + 256) % 256;
                        }
                        if next_scale != 0 {
                            last_scale = next_scale;
                        }
                    }
                }
            }
        }
    } else {
        s.chroma_format_idc = 1;
    }

    s.log2_max_frame_num_minus4 = r.read_ue() as u8;
    s.pic_order_cnt_type = r.read_ue() as u8;
    match s.pic_order_cnt_type {
        0 => {
            s.log2_max_pic_order_cnt_lsb_minus4 = r.read_ue() as u8;
        }
        1 => {
            // delta_pic_order_always_zero_flag
            r.read_bit();
            // offset_for_non_ref_pic
            r.read_se();
            // offset_for_top_to_bottom_field
            r.read_se();
            let n = r.read_ue();
            for _ in 0..n.min(256) {
                r.read_se();
            }
        }
        _ => {}
    }
    s.max_num_ref_frames = r.read_ue() as u8;
    s.gaps_in_frame_num_value_allowed_flag = r.read_bit();
    s.pic_width_in_mbs_minus1 = r.read_ue();
    s.pic_height_in_map_units_minus1 = r.read_ue();
    s.frame_mbs_only_flag = r.read_bit();
    if s.frame_mbs_only_flag == 0 {
        s.mb_adaptive_frame_field_flag = r.read_bit();
    }
    s.direct_8x8_inference_flag = r.read_bit();
    s.frame_cropping_flag = r.read_bit();
    if s.frame_cropping_flag == 1 {
        s.frame_crop_left_offset = r.read_ue();
        s.frame_crop_right_offset = r.read_ue();
        s.frame_crop_top_offset = r.read_ue();
        s.frame_crop_bottom_offset = r.read_ue();
    }
    s.vui_parameters_present_flag = r.read_bit();

    // Compute coded dimensions.
    let mb_w = (s.pic_width_in_mbs_minus1 + 1) * 16;
    let mb_h = (s.pic_height_in_map_units_minus1 + 1)
        * 16
        * if s.frame_mbs_only_flag != 0 { 1 } else { 2 };
    s.raw_width = mb_w;
    s.raw_height = mb_h;

    // Cropping units depend on chroma_format_idc.
    let (crop_unit_x, crop_unit_y) = match s.chroma_format_idc {
        0 => (1, 2 - s.frame_mbs_only_flag as u32),
        1 => (2, 2 * (2 - s.frame_mbs_only_flag as u32)),
        2 => (2, 1 * (2 - s.frame_mbs_only_flag as u32)),
        3 => (1, 1 * (2 - s.frame_mbs_only_flag as u32)),
        _ => (1, 1),
    };
    let cl = s.frame_crop_left_offset * crop_unit_x;
    let cr = s.frame_crop_right_offset * crop_unit_x;
    let ct = s.frame_crop_top_offset * crop_unit_y;
    let cb = s.frame_crop_bottom_offset * crop_unit_y;
    s.coded_width = mb_w.saturating_sub(cl + cr);
    s.coded_height = mb_h.saturating_sub(ct + cb);

    Some(s)
}

/// Decoded subset of `StdVideoH264PictureParameterSet`.
#[derive(Debug, Clone, Default)]
pub struct ParsedPps {
    pub pic_parameter_set_id: u8,
    pub seq_parameter_set_id: u8,
    pub entropy_coding_mode_flag: u8,
    pub bottom_field_pic_order_in_frame_present_flag: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub weighted_pred_flag: u8,
    pub weighted_bipred_idc: u8,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub deblocking_filter_control_present_flag: u8,
    pub constrained_intra_pred_flag: u8,
    pub redundant_pic_cnt_present_flag: u8,
    pub transform_8x8_mode_flag: u8,
    pub pic_scaling_matrix_present_flag: u8,
    pub second_chroma_qp_index_offset: i8,
}

/// Parse a PPS NAL.
pub fn parse_pps(nal: &[u8]) -> Option<ParsedPps> {
    if nal.is_empty() || (nal[0] & 0x1F) != nal_type::PPS {
        return None;
    }
    let rbsp = rbsp_strip(&nal[1..]);
    let mut r = BitReader::new(&rbsp);

    let mut p = ParsedPps::default();
    p.pic_parameter_set_id = r.read_ue() as u8;
    p.seq_parameter_set_id = r.read_ue() as u8;
    p.entropy_coding_mode_flag = r.read_bit();
    p.bottom_field_pic_order_in_frame_present_flag = r.read_bit();
    let num_slice_groups = r.read_ue() + 1;
    if num_slice_groups > 1 {
        // Skip slice-group map decoding — IDR test stream has 1 slice group.
        return Some(p);
    }
    p.num_ref_idx_l0_default_active_minus1 = r.read_ue() as u8;
    p.num_ref_idx_l1_default_active_minus1 = r.read_ue() as u8;
    p.weighted_pred_flag = r.read_bit();
    p.weighted_bipred_idc = r.read_bits(2) as u8;
    p.pic_init_qp_minus26 = r.read_se() as i8;
    p.pic_init_qs_minus26 = r.read_se() as i8;
    p.chroma_qp_index_offset = r.read_se() as i8;
    p.deblocking_filter_control_present_flag = r.read_bit();
    p.constrained_intra_pred_flag = r.read_bit();
    p.redundant_pic_cnt_present_flag = r.read_bit();
    p.second_chroma_qp_index_offset = p.chroma_qp_index_offset;
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nal_iteration_finds_three_units() {
        // 00 00 00 01 65 ff 00 00 00 01 67 aa 00 00 01 68 bb
        // -> three NAL units of types 0x65, 0x67, 0x68.
        let buf = [
            0x00, 0x00, 0x00, 0x01, 0x65, 0xff, 0x00, 0x00, 0x00, 0x01, 0x67, 0xaa, 0x00, 0x00,
            0x01, 0x68, 0xbb,
        ];
        let nals = iter_nals(&buf);
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0][0], 0x65);
        assert_eq!(nals[1][0], 0x67);
        assert_eq!(nals[2][0], 0x68);
    }

    #[test]
    fn rbsp_strips_emulation_byte() {
        // 00 00 03 00 -> 00 00 00
        let v = rbsp_strip(&[0x00, 0x00, 0x03, 0x00]);
        assert_eq!(v, vec![0x00, 0x00, 0x00]);
    }

    #[test]
    fn ue_basic() {
        // 1 -> 0, 010 -> 1, 011 -> 2, 00100 -> 3, 00101 -> 4
        let buf = [0b10100110, 0b00100001, 0b01000000];
        let mut r = BitReader::new(&buf);
        assert_eq!(r.read_ue(), 0);
        assert_eq!(r.read_ue(), 1);
        assert_eq!(r.read_ue(), 2);
    }
}
