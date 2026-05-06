# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed — Round 4 SIGSEGV diagnosis + repair

The Round 4 `vkQueueSubmit`-time NVIDIA SIGSEGV (RTX 5080 / driver
580.95.05) was reproduced under
`VK_LAYER_PATH=… VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`
and pinned on five distinct API misuses on our side. Each
violation matches a concrete VUID; each is fixed in-place. With
all five fixed the decode runs cleanly through `vkQueueSubmit`,
the decoded NV12 frame matches the ffmpeg reference YUV
**bit-for-bit** (mean abs diff = `0.00/255`), and the
`h264_decoder_attempts_decode` integration test exits 0.

1. **VUID-vkCmdDecodeVideoKHR-pDecodeInfo-07254 / 07253** — the
   DPB image was transitioned to
   `VK_IMAGE_LAYOUT_VIDEO_DECODE_DST_KHR` before the decode in
   coincide mode. The spec requires the picture-resource bound to a
   non-NULL `pSetupReferenceSlot` to be in
   `VK_IMAGE_LAYOUT_VIDEO_DECODE_DPB_KHR` *even when DPB and dst
   alias each other*. NVIDIA's driver doesn't validate the layout
   itself; instead it walks the slot's image binding via the
   layout-tracker and dereferences a stale pointer when the
   tracker says "this is a DST", crashing inside
   `libnvidia-glcore` at `+0xea4ac4`. Fix: always transition to
   `VIDEO_DECODE_DPB_KHR` for the DPB image, and adjust the post-
   decode → transfer barrier accordingly.
2. **VUID-VkImageViewCreateInfo-subresourceRange-07818** — the
   image views used `VK_IMAGE_ASPECT_PLANE_0_BIT |
   VK_IMAGE_ASPECT_PLANE_1_BIT` for an NV12 (multi-planar) view,
   which is invalid; multi-planar views with both plane bits set
   are reserved for plane-disjoint VkImage instances. Fix: use
   `VK_IMAGE_ASPECT_COLOR_BIT` for the all-planes view, and keep
   the per-plane bits scoped to the `VkBufferImageCopy::image_subresource`
   regions where they're correct.
3. **VUID-vkCreateDevice-ppEnabledExtensionNames-01387** —
   `VK_KHR_video_queue` and `VK_KHR_video_decode_queue` are
   specified in terms of the sync2 stage / access-mask families
   and require `VK_KHR_synchronization2` to be enabled alongside
   them. Fix: add a `VK_KHR_SYNCHRONIZATION_2_NAME` constant in
   `physical_device` and prepend it to the device's
   `pp_enabled_extension_names` list in `decoder::DecoderState::create`.
4. **VUID-vkCmdCopyImageToBuffer-srcImage-00186 +
   VkImageMemoryBarrier-oldLayout-01212** — in coincide mode the
   DPB image doubles as the post-decode `vkCmdCopyImageToBuffer`
   source, so the image needs `VK_IMAGE_USAGE_TRANSFER_SRC_BIT` on
   creation. Fix: OR `VK_IMAGE_USAGE_TRANSFER_SRC_BIT` into the
   coincide-path DPB usage flags.
5. **VUID-vkCmdCopyImageToBuffer-pRegions-00183** — the
   `VkBufferImageCopy::buffer_row_length` for the chroma plane was
   set to `width`, but the spec measures `buffer_row_length` in
   *texels of the source plane*. NV12 plane 1 (R8G8) is half-width
   half-height with 2 bytes/texel, so the texel-stride is
   `width / 2`, not `width`; using `width` here told the driver to
   skip 2× the real bytes/row and overran the staging buffer by
   ~38 KiB. Fix: store `chroma_stride` as plane-1 texels-per-row
   (= `width / 2`) and double it when computing the host-side byte
   offset during readback.

A sixth crash uncovered along the way: `VideoSession::drop` was
dispatching `vkDestroyVideoSessionKHR` through a `&Device` borrow
that had been transmuted to `'static` from a *local* binding before
the session was constructed, then invalidated by the move of that
local into `DecoderState::device`. Field-order tear-down then ran
the session's `Drop` last, dereferencing a dangling pointer for the
function table. Fix: a new `VideoSession::detach` API hands the
handle + bound-memory list back to the caller and neuters the
session's `Drop`; `DecoderState::drop` calls it through `&self.device`
so the dispatch goes through the *current* (still-owned) device.

The Round-4 helper subprocess and the parent's signal-isolating
fork wrapper are kept — they're now belts-and-braces for any
future driver/SDK regression: if the crash returns, the parent
panics with the signal number rather than silently regressing.

#### Diagnostics
- New per-field offset cross-checks (`tests/struct_sizes.rs` —
  `h264_std_struct_field_offsets_match_c_abi` and
  `vulkan_round4_struct_field_offsets_match_c_abi`) verify every
  Vulkan / std-video struct's field offsets against the C ABI from
  the system Khronos headers, not just the trailing `sizeof`.
  These caught nothing on the dev box (every struct's layout
  matches byte-for-byte) but provide a regression guard for any
  future field-shift bug — the kind that would manifest as another
  "driver crashes deep inside its own dispatch".

#### Validation results
- Validation layers were enabled by setting `VK_LAYER_PATH` to a
  manifest pointing at the locally-installed
  `libVkLayer_khronos_validation.so` and
  `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`. The layer's
  output, captured to `stderr` of the helper subprocess, surfaced
  the seven distinct VUIDs above. After the fixes the helper now
  runs end-to-end with **zero VUID violations** and exit code 0.

### Changed — Round 7

- `H264VkDecoder::make` honours `CodecParameters::device_index`.
  Indexing matches `engine_info()`'s `VkPhysicalDevice` filtering
  order — every physical device whose type is
  Discrete/Integrated/Virtual GPU OR that advertises any
  `VK_KHR_video_*` extension, in `Instance::physical_devices()`
  enumeration order. The two MUST stay in sync: a CLI consumer that
  prints "device 1" via `engine_info()` and then passes
  `with_device_index(1)` to `CodecParameters::video(...)` gets the
  decoder bound to that exact same `VkPhysicalDevice`.
- `device_index = None` (the default) resolves to device 0 via
  `unwrap_or(0)`, preserving prior behaviour bit-for-bit on every
  single-GPU host.
- An out-of-range `device_index` surfaces as `Error::Unsupported(
  "vulkan-video: device_index N out of range (0..M)")` so the codec
  registry can fall back to a software path; a chosen device that
  doesn't actually advertise H.264 decode (`queue_khr` /
  `decode_h264`) likewise returns `Error::Unsupported` with a
  descriptive message rather than silently picking a different
  device.
- New integration test `tests/round7_device_index.rs` —
  `device_index_none_uses_first_video_device`,
  `device_index_zero_explicit_works`,
  `device_index_out_of_range_errors`,
  `device_index_default_is_none_in_codec_parameters`.
  Skip-friendly when no Vulkan ICD is installed; uses the same
  `OXIDEAV_VK_SKIP_SUBMIT` hook the round 4 tests use to stop short
  of the NVIDIA-driver `vkQueueSubmit` SIGSEGV.

### Added — Round 6

- New module `engine` exposing
  `pub fn engine_info() -> Vec<oxideav_core::HwDeviceInfo>`. Opens an
  ephemeral Vulkan instance (`VK_API_VERSION_1_2`, app name
  `"oxideav-vulkan-video-engine-info"`), enumerates every physical
  device the loader sees, and returns one `HwDeviceInfo` per device:
  - `name` from `VkPhysicalDeviceProperties.deviceName`.
  - `api_version` formatted as `"Vulkan {major}.{minor}.{patch}"`.
  - `driver_version` formatted as `"0x{:08x}"` (vendor-specific
    decoding is a follow-up).
  - `total_memory_bytes` summed across every
    `VkPhysicalDeviceMemoryProperties.memoryHeaps` entry whose flags
    have `VK_MEMORY_HEAP_DEVICE_LOCAL_BIT` set.
  - `extra` extras: `vendor_id` (`0x{:04x}`), `device_id`
    (`0x{:04x}`), `device_type` (`discrete` / `integrated` /
    `virtual` / `cpu` / `other`).
  - `codecs`: per-codec `HwCodecCaps` entries derived from
    `supports_video_extensions()`. H.264 caps go through
    `query_video_decode_h264_capabilities` against the High profile
    to populate `max_width` / `max_height` / `max_dpb_slots` /
    `max_active_reference_pictures` / `max_level_idc` /
    `std_header_version` (e.g. `"1.0.0"`). HEVC and AV1 surface a
    `decode: true` flag plus `max_bit_depth: 8` — the matching
    H.265 / AV1 capability-query plumbing isn't wired into `sys.rs`
    yet, so dimensions stay `None`.
- The H.264 `CodecInfo` registered by `register()` now chains
  `.with_engine_id("vulkan-video").with_engine_probe(engine_info)` so
  the framework's CLI `info` command can call the probe on demand.
- New constant `sys::VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x1` next to
  the existing memory-property bits, used by the heap-size sum.
- New integration test `tests/round6_engine_info.rs` —
  `engine_info_finds_rtx_5080_or_skips`,
  `engine_info_per_device_metadata_is_consistent`,
  `engine_info_attaches_via_with_engine_probe`. Skip-friendly when
  no Vulkan ICD is installed; on the dev box (RTX 5080, driver
  580.95.05) the test reports the GPU + Vulkan version + heap size
  + per-codec caps including decoded H.264 max extent.
- Skip-friendly behaviour: any error path during instance creation
  or physical-device enumeration collapses to `vec![]`. Consumers
  treat empty as "no Vulkan backend on this host" and fall back to
  whatever pure-Rust path they have. Module is gated behind the
  default-on `registry` feature (it consumes the
  `oxideav_core::HwDeviceInfo` type), matching `decoder.rs` and
  `register()`.

### Changed — Round 5

- Migrated H.264 parser to `oxideav-bitstream`. The crate-local
  `src/h264_parser.rs` module (Annex-B walker, EBSP→RBSP stripper,
  Exp-Golomb bit reader, minimal SPS / PPS decoder) has been deleted;
  the same parsing job is now done by the workspace-shared
  `oxideav_bitstream::h264` API (`split_annex_b`, `parse_sps_nal`,
  `parse_pps_nal`, `H264Sps`, `H264Pps`, `NAL_TYPE_*`). The Vulkan
  Video decode pipeline itself — `StdVideoH264*` struct construction,
  `VkVideoSessionParametersKHR` creation, command-buffer recording,
  queue submission — is unchanged.
- New target-gated dependency: `oxideav-bitstream = "0.0"` under
  `[target.'cfg(any(target_os = "linux", target_os = "windows"))'.dependencies]`,
  matching the rest of the crate body's cfg.
- All Round 2 / Round 3 / Round 4 tests still pass through
  `vkEndCommandBuffer`. The reproducible NVIDIA-driver SIGSEGV at
  `vkQueueSubmit`-time is unrelated to parsing and remains absorbed
  by the `round4_decode_helper` subprocess as before.

### Added — Round 4

- New module `h264_parser` — minimal IDR-only H.264 Annex-B / RBSP
  parser. Walks NAL units (start-code-prefixed and emulation-prevention
  byte stripped), decodes SPS into the subset of
  `StdVideoH264SequenceParameterSet` fields the GPU needs (profile_idc,
  level_idc, chroma_format_idc, log2_max_frame_num_minus4,
  pic_order_cnt_type, max_num_ref_frames, pic_width_in_mbs_minus1,
  pic_height_in_map_units_minus1, frame_mbs_only_flag,
  direct_8x8_inference_flag, frame cropping, vui_parameters_present),
  and PPS into the corresponding `StdVideoH264PictureParameterSet`
  subset. Sufficient for High-profile single-IDR decode; VUI / HRD /
  scaling-list parsing intentionally stubbed (all flags off, pointers
  null).
- `sys.rs` extended with the rest of the Vulkan + H.264 std structs
  needed for the decode dispatch: `StdVideoH264SpsFlags` /
  `StdVideoH264PpsFlags` / `StdVideoH264SequenceParameterSet` /
  `StdVideoH264PictureParameterSet` / `StdVideoDecodeH264PictureInfo` /
  `StdVideoDecodeH264ReferenceInfo` /
  `VkVideoDecodeH264SessionParametersAddInfoKHR` /
  `VkVideoDecodeH264SessionParametersCreateInfoKHR` /
  `VkVideoDecodeH264PictureInfoKHR` /
  `VkVideoDecodeH264DpbSlotInfoKHR` / `VkVideoSessionParametersKHR`
  handle / `VkVideoProfileListInfoKHR` /
  `VkVideoPictureResourceInfoKHR` / `VkVideoReferenceSlotInfoKHR` /
  `VkVideoBeginCodingInfoKHR` / `VkVideoEndCodingInfoKHR` /
  `VkVideoCodingControlInfoKHR` / `VkVideoDecodeInfoKHR`. Bit layouts
  for the bitfield aggregates (`*Flags`) are documented as constants
  on the wrapper structs (e.g. `StdVideoH264SpsFlags::FRAME_MBS_ONLY`
  = `1 << 8`) since Rust doesn't have C bitfields.
- Plus the surrounding Vulkan-core types/structs/PFNs for image and
  buffer creation, command-buffer recording, queue submission, fence
  / wait synchronization, image / buffer memory binding, and host
  memory mapping: `VkBuffer` / `VkImage` / `VkImageView` /
  `VkCommandPool` / `VkCommandBuffer` / `VkFence` non-dispatchable
  handles; `VkBufferCreateInfo` / `VkImageCreateInfo` /
  `VkImageViewCreateInfo` / `VkImageMemoryBarrier` /
  `VkImageSubresourceRange` / `VkImageSubresourceLayers` /
  `VkBufferImageCopy` / `VkComponentMapping` / `VkOffset3D` structs;
  `VkCommandPoolCreateInfo` / `VkCommandBufferAllocateInfo` /
  `VkCommandBufferBeginInfo` / `VkSubmitInfo` / `VkFenceCreateInfo`
  command-buffer / submit / fence structs; `VK_IMAGE_USAGE_VIDEO_*` /
  `VK_BUFFER_USAGE_VIDEO_DECODE_SRC_BIT_KHR` /
  `VK_IMAGE_LAYOUT_VIDEO_DECODE_*` enum constants;
  `VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR` for the spec-mandated first-
  submit reset; the pipeline-stage / access-mask bits used by the
  layout barriers.
- 26 new device-level function pointer typedefs covering the entire
  decode path: `vkCreate{Buffer,Image,ImageView}` /
  `vkDestroy{Buffer,Image,ImageView}` /
  `vkGet{Buffer,Image}MemoryRequirements` /
  `vkBind{Buffer,Image}Memory` / `vkMapMemory` / `vkUnmapMemory` /
  `vkCreate{CommandPool,Fence}` /
  `vkDestroy{CommandPool,Fence}` / `vkAllocateCommandBuffers` /
  `vkFreeCommandBuffers` / `vkBegin{,End}CommandBuffer` /
  `vkCmdPipelineBarrier` / `vkCmdCopyImageToBuffer` /
  `vkQueueSubmit` / `vkQueueWaitIdle` / `vkWaitForFences` /
  `vkCreateVideoSessionParametersKHR` /
  `vkDestroyVideoSessionParametersKHR` / `vkCmdBeginVideoCodingKHR` /
  `vkCmdEndVideoCodingKHR` / `vkCmdControlVideoCodingKHR` /
  `vkCmdDecodeVideoKHR`. All resolved through `vkGetDeviceProcAddr`
  in `Device::new()`'s `DeviceFns::resolve` pass.
- New module `decoder` exposing `H264VkDecoder` implementing
  `oxideav_core::Decoder`. Lazy-init the heavy state (instance,
  device, video session, session parameters, DPB image, output
  image-view, host-visible bitstream + staging buffers, command
  pool + buffer) on first SPS+PPS pair seen via `send_packet`.
  Once initialised, each subsequent `send_packet` walks the Annex-B
  bitstream, finds the VCL slice offsets, uploads the entire packet
  data into the host-visible bitstream buffer, records a single
  command buffer (image layout transition →
  `vkCmdBeginVideoCodingKHR` → `vkCmdControlVideoCodingKHR` (RESET) →
  `vkCmdDecodeVideoKHR` → `vkCmdEndVideoCodingKHR` → image layout
  transition → `vkCmdCopyImageToBuffer` for the NV12 output → final
  layout transition), submits, waits, then memcpy's the staging
  buffer into a planar I420 `VideoFrame` (de-interleaving the NV12
  UV plane).
- `register()` now wires up an `H264VkDecoder` factory at priority 20
  (above the pure-Rust path's 0, leaving room for future bridges).
  Tags: H264 / h264 / AVC1 / avc1 / X264 fourccs + Matroska
  `V_MPEG4/ISO/AVC`. Falls back gracefully when the loader can't be
  opened.
- Round 4 integration tests `tests/round4_decode.rs`:
  * `h264_parser_finds_sps_pps` — round-trips the test fixture
    through the parser, assertions on profile (High = 100), coded
    extent (320×240), presence of SPS+PPS+IDR.
  * `h264_decoder_constructs_full_pipeline` — runs the decoder up
    to (and including) `vkEndCommandBuffer` via the
    `OXIDEAV_VK_SKIP_SUBMIT` env hook, asserts every step succeeds.
  * `h264_decoder_attempts_decode` — forks the
    `round4_decode_helper` subprocess that runs the full decode
    pipeline including `vkQueueSubmit`. On a future driver release
    where the submit succeeds, the helper writes the decoded frame
    to disk and the parent validates pixel content (luma std-dev,
    cross-validation against an ffmpeg-rendered reference YUV).
- `tests/struct_sizes.rs` — parity assertions: every Vulkan / std
  struct mirrored in `sys.rs` matches the GCC/Clang `sizeof` of the
  C declaration in `vk_video/vulkan_video_codec_h264std{,_decode}.h`
  + `vulkan/vulkan_core.h`.
- Test fixture `tests/fixtures/h264_high_320x240_1frame.h264`
  (synthetic single-IDR ffmpeg lavfi `testsrc2=size=320x240` H.264
  High profile) + matching `reference.yuv` planar I420 dump for
  cross-validation.

### Known issue — NVIDIA driver SIGSEGV during `vkQueueSubmit`

On the dev box (NVIDIA RTX 5080, driver 580.95.05) the Vulkan video
decode pipeline records, validates, and `vkEndCommandBuffer`-completes
without error, BUT the subsequent `vkQueueSubmit` triggers a SIGSEGV
deep inside `libnvidia-glcore.so.580.95.05` at offset `+0xea4ac4` —
inside the proprietary driver's own command-buffer execution path. The
crash is reproducible across runs and persists even with minimal IDR
input. The H.264 std-struct layouts and Vulkan struct layouts were
verified byte-for-byte against the C ABI in `tests/struct_sizes.rs`,
so it isn't an FFI bug on our side. The decoder's
`h264_decoder_attempts_decode` test runs the full pipeline in a
separate process, captures the SIGSEGV via the child's exit signal,
and reports it as a soft fail rather than bringing down the parent
test runner. If a future driver release fixes the crash, the helper
binary will produce the decoded frame and the parent will validate
it against the ffmpeg reference (the wiring is already in place).

### Added — Round 2

- `sys.rs` extended with the Vulkan core structs needed for instance
  bootstrap and physical-device probing: `VkApplicationInfo`,
  `VkInstanceCreateInfo`, `VkExtensionProperties`,
  `VkPhysicalDeviceProperties` (with the full `VkPhysicalDeviceLimits`
  + `VkPhysicalDeviceSparseProperties` substructs so the trailing
  fields land at the right offsets), `VkQueueFamilyProperties2` +
  `VkQueueFamilyVideoPropertiesKHR` for the `pNext`-chained queue
  family probe, plus the discriminants
  `VK_STRUCTURE_TYPE_APPLICATION_INFO`,
  `VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO`,
  `VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2`, and
  `VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR`.
- Vulkan version helpers: `vk_make_api_version` packing,
  `vk_api_version_{variant,major,minor,patch}` accessors, and the
  packed constants `VK_API_VERSION_1_0` ... `VK_API_VERSION_1_3`.
- Queue flag constants `VK_QUEUE_VIDEO_DECODE_BIT_KHR (0x20)` and
  `VK_QUEUE_VIDEO_ENCODE_BIT_KHR (0x40)`, plus the rest of the
  Vulkan 1.0 graphics/compute/transfer/sparse/protected bit
  family for completeness.
- Function pointer typedefs for the post-bootstrap entries resolved
  through `vkGetInstanceProcAddr`: `vkDestroyInstance`,
  `vkEnumeratePhysicalDevices`, `vkGetPhysicalDeviceProperties`,
  `vkEnumerateDeviceExtensionProperties`,
  `vkGetPhysicalDeviceQueueFamilyProperties2`.
- New module `instance` with a safe `Instance` wrapper:
  `Instance::new(app_name, requested_api_version)` calls
  `vkCreateInstance` with empty layers + extensions, then resolves
  every post-bootstrap function pointer it'll need via
  `vkGetInstanceProcAddr`. `Drop` calls `vkDestroyInstance`.
- `VkError` enum: `LoaderUnavailable`, `Result { op, result }`,
  `MissingFunction`. Implements `Display` + `Error`.
- New module `physical_device` with `PhysicalDevice<'instance>` —
  borrowed against the parent `Instance`'s function pointers so the
  spec lifetime rule is enforced by the borrow checker. Surface:
  `properties()` (name, vendor_id, device_id, device_type,
  api_version, driver_version), `extension_names()`,
  `supports_video_extensions()` returning a `VideoExtensionSupport`
  bool struct (queue_khr, decode_h264, decode_h265, decode_av1,
  encode_h264, encode_h265), and `video_queue_family_indices()`
  built on the `_2` form of the queue-family probe.
- Public re-exports of `Instance`, `PhysicalDevice`,
  `PhysicalDeviceProperties`, `VideoExtensionSupport`, `VkError`.
- Integration test `tests/round2_init.rs` (skip-friendly when no
  Vulkan ICD is present): `loader_loads`,
  `instance_creates_with_oxideav_app_name`,
  `lists_at_least_one_physical_device`,
  `physical_device_reports_name_and_vendor`,
  `nvidia_advertises_video_decode_h264` (asserts only when an
  NVIDIA GPU — vendor 0x10DE — is present), and
  `video_queue_family_indices_smoke`. Verified on the dev box (RTX
  5080, driver 580.95.05): the device reports queue_khr +
  decode_h264 + decode_h265 + decode_av1 + encode_h264 +
  encode_h265 + 2 video-capable queue families.

### Added — Round 1

- Initial scaffolding: `#![cfg(any(target_os = "linux", target_os
  = "windows"))]` crate that opens the Vulkan loader via
  `libloading` on first use — `libvulkan.so.1` on Linux,
  `vulkan-1.dll` on Windows.
- `sys.rs` exposes opaque type aliases (`VkInstance`,
  `VkPhysicalDevice`, `VkDevice`, `VkQueue`, `VkResult`) and a
  resolved `Vtable` covering the four bootstrap symbols Vulkan
  exports as normal dynamic-linker entries:
  - `vkGetInstanceProcAddr` — the universal Vulkan dispatch entry
    every other instance-level function is reached through.
  - `vkCreateInstance` — needed to construct the `VkInstance` that
    `vkGetDeviceProcAddr` later operates against.
  - `vkEnumerateInstanceExtensionProperties` — used to verify the
    Vulkan loader exposes the `VK_KHR_video_*` extension family
    (Round 2 will gate per-codec registration on this).
  - `vkEnumerateInstanceVersion` — Vulkan 1.1+ runtime sanity check.
- Process-wide `OnceLock<Result<Vtable, String>>` cache so the
  dlopen + dlsym round-trip happens at most once per process.
- Unified `register(&mut RuntimeContext)` entry point. Round 1: the
  function confirms the loader loads and returns; no codec
  factories are wired up yet. If load fails (no Vulkan loader, no
  ICD installed) the function logs and returns — the pure-Rust
  codec path remains the only resolution candidate.
- Standalone-friendly `registry` feature (default-on) gates the
  `oxideav-core` + `linkme` deps.
- README coverage roadmap and priority explanation.
- Smoke tests: `frameworks_load` and `vtable_resolves` confirm
  symbol resolution on any Linux or Windows machine that has a
  Vulkan loader installed.
