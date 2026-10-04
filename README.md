# oxideav-vulkan-video

[![CI](https://github.com/OxideAV/oxideav-vulkan-video/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-vulkan-video/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-vulkan-video.svg)](https://crates.io/crates/oxideav-vulkan-video) [![docs.rs](https://docs.rs/oxideav-vulkan-video/badge.svg)](https://docs.rs/oxideav-vulkan-video) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Vulkan Video hardware decode/encode bridge for the
[oxideav](https://github.com/OxideAV/oxideav) framework. Builds on
**Linux and Windows**.

## Why a bridge crate?

The Vulkan Video extension family (`VK_KHR_video_queue`,
`VK_KHR_video_decode_h264`, `VK_KHR_video_decode_h265`,
`VK_KHR_video_decode_av1`, `VK_KHR_video_encode_*`) is the vendor- and
OS-neutral path for HW acceleration. Unlike VA-API (Linux-only) and
NVENC (single-vendor), Vulkan Video is implemented in the Vulkan ICD
layer itself and ships across all three major GPU vendors on both Linux
and Windows. Decode is widely available today; encode is rolling out.

This crate is a **thin runtime-loaded bridge** — no compile-time link
dependency on the Vulkan loader or any vendor ICD. The loader is opened
via [`libloading`] on first use:

| Platform | Loader filename |
|----------|-----------------|
| Linux    | `libvulkan.so.1` |
| Windows  | `vulkan-1.dll`   |

On Windows the loader is installed by the Vulkan SDK, by GPU driver
packages, and by recent Windows builds.

## Programming model

Only `vkGetInstanceProcAddr` is meaningfully resolved by `dlsym`. Every
other Vulkan function — including all video extension entry points
(`vkCmdBeginVideoCodingKHR`, `vkGetVideoSessionMemoryRequirementsKHR`,
…) — is reached via `vkGetInstanceProcAddr` (instance-level) or
`vkGetDeviceProcAddr` (device-level) after a `VkInstance` is created. So
the crate's bootstrap vtable is intentionally tiny:

* `vkGetInstanceProcAddr`
* `vkCreateInstance`
* `vkEnumerateInstanceExtensionProperties`
* `vkEnumerateInstanceVersion`

These construct a `VkInstance`, enumerate physical devices, and probe
the `VK_KHR_video_*` extension family. Every other Vulkan entry is
resolved on demand.

## Importing an existing device

If your application already owns a Vulkan device (a renderer, a
compute pipeline, …), decode can share it instead of the crate creating
its own instance / device:

```rust,ignore
use oxideav_vulkan_video::{decoder::H264VkDecoder, ExternalDevice};

let ext = ExternalDevice::new(vk_instance, vk_physical_device, vk_device, video_qfi);
// SAFETY: handles are valid, outlive the decoder, video extensions
// were enabled at vkCreateDevice time; see the method docs.
let mut dec = unsafe { H264VkDecoder::make_with_device(&params, ext) }?;
```

The imported handles are wrapped **non-owning** — nothing in the
decoder's `Drop` destroys your instance or device; only the objects the
crate created on top of them (video session, images, buffers, command
pool) are torn down. `ExternalDevice::with_queue_index` selects a queue
other than 0 within the family, and `with_get_instance_proc_addr`
supports non-standard loaders. The device must have been created with
`VK_KHR_video_queue`, `VK_KHR_video_decode_queue`,
`VK_KHR_video_decode_h264`, and (below Vulkan 1.3)
`VK_KHR_synchronization2` enabled.

Lower-level non-owning wrappers (`Instance::from_raw`,
`Device::from_raw`, …) are available without the `registry` feature for
tooling that only wants the raw bridge.

## Fallback behaviour

Two distinct failure paths fall back automatically to the pure-Rust
codec:

1. **Load failure** — no Vulkan loader / ICD on the system. `register()`
   logs and returns without registering.
2. **Init failure** — `vkCreateInstance` succeeds but no
   `VkPhysicalDevice` advertises the requested `VK_KHR_video_*`
   extension, or the video-decode/encode queue family is missing. The
   factory returns `Err`; `oxideav_pipeline::make_decoder_with` /
   `make_encoder_with` (the selection walker over the registry) falls
   back to the next-priority impl.

Pipelines that **require** hardware opt out of the SW fallback by
setting `CodecPreferences { require_hardware: true, .. }`.

## Platform gating

The whole crate is `#![cfg(any(target_os = "linux", target_os =
"windows"))]`. On macOS it compiles to an empty rlib; `oxideav-meta` only
pulls the crate in (and so only calls its `register`) under the same
cfg. (Vulkan
is reachable on macOS via MoltenVK with a different loading story — out
of scope.)

## Priority and opt-out

The H.264 decoder registers at `H264_DECODE_PRIORITY` (150; lower
wins) — *behind* the pure-Rust `h264_sw` (100), because it still
decodes IDR pictures only. Callers opt in with
`CodecPreferences::prefer` / `require_hardware`. `--no-hwaccel` on
the `oxideav` CLI biases dispatch away from HW factories without
unregistering them.

## Coverage

| Codec | Decode | Encode |
|-------|--------|--------|
| H.264 | Full pipeline wired end-to-end (session + parameters, DPB NV12 image, bitstream/staging buffers, command recording, NV12 readback); validation-clean | planned |
| HEVC  | Capability query wired; session/decode pipeline planned | planned |
| AV1   | Capability query wired; session/decode pipeline planned | planned |
| VP9   | — | — |

Capability queries (`query_video_decode_h264/h265/av1_capabilities`)
chain the per-codec `VkVideoProfileInfoKHR` / `VkVideoCapabilitiesKHR`
structures and populate `engine_info()` rows with max dimensions, DPB
slots, reference-picture counts, level, and canonical profile/level
labels. Struct sizes and field offsets are cross-checked against the C
ABI in `tests/struct_sizes.rs`; integration tests skip gracefully when
no Vulkan ICD or per-codec extension is present.

## Workspace policy

Calling a system OS / driver API via FFI is the same shape as calling
`libc::malloc` — it's the platform, not a copied algorithm. The
workspace's clean-room rule (no embedding source from external codec
libraries) does not apply to this crate.

## License

MIT.
