# oxideav-vulkan-video

Linux Vulkan Video hardware decode/encode bridge for the [oxideav](https://github.com/OxideAV/oxideav) framework.

## Why a bridge crate?

The Vulkan Video extension family (`VK_KHR_video_queue`, `VK_KHR_video_decode_h264`, `VK_KHR_video_decode_h265`, `VK_KHR_video_decode_av1`, `VK_KHR_video_encode_*`) is the **vendor-neutral** path forward for HW acceleration on Linux. Unlike VA-API (Intel/AMD-leaning) and NVENC (NVIDIA-only), Vulkan Video is implemented in the Vulkan ICD layer itself and is gradually shipping across all three major GPU vendors. As of 2025, decode is widely available; encode is rolling out.

This crate is a **thin runtime-loaded bridge** — no compile-time link dependency on the Vulkan loader or any vendor ICD. `libvulkan.so.1` is opened via [`libloading`] on first use.

## Programming model

Vulkan is unusual in that **only `vkGetInstanceProcAddr` is meaningfully resolved by `dlsym`**. Every other Vulkan function — including all video extension entry points (`vkCmdBeginVideoCodingKHR`, `vkGetVideoSessionMemoryRequirementsKHR`, …) — is reached via `vkGetInstanceProcAddr` (instance-level entries) or `vkGetDeviceProcAddr` (device-level entries) after a `VkInstance` is created. So this crate's bootstrap vtable is intentionally tiny:

* `vkGetInstanceProcAddr`
* `vkCreateInstance`
* `vkEnumerateInstanceExtensionProperties`
* `vkEnumerateInstanceVersion`

Round 2 will use these to construct a `VkInstance`, query for `VK_KHR_video_*` extension support, and resolve every other Vulkan entry through `vkGetInstanceProcAddr` / `vkGetDeviceProcAddr`.

## Fallback behaviour

Two distinct failure paths fall back automatically to the pure-Rust codec:

1. **Load failure** — `libvulkan.so.1` not installed, no Vulkan ICD on the system (e.g. headless CI without Mesa or NVIDIA driver). `register()` logs and returns without registering.
2. **Init failure** — `vkCreateInstance` succeeds but `vkEnumerateDeviceExtensionProperties` reports the requested `VK_KHR_video_*` extension is unsupported by every available `VkPhysicalDevice`, or the queue family for video-decode/encode operations is missing. The factory returns `Err`; the registry falls back to the next-priority impl.

Pipelines that **require** hardware can opt out of the SW fallback by setting `CodecPreferences { require_hardware: true, .. }`.

## Platform gating

The whole crate is `#![cfg(target_os = "linux")]`. On macOS / Windows it compiles to an empty rlib; the umbrella `oxideav` crate gates the `register` call behind the same cfg. (Vulkan Video is also available on Windows + macOS via MoltenVK, but cross-platform support is out of scope for Round 1.)

## Priority

Hardware factories register with `CodecCapabilities::with_priority(20)` — slightly higher (worse) than VA-API's 10 and NVENC's 5, because Vulkan Video drivers are still maturing and the per-vendor implementation quality varies. As stability improves we will lower the priority number.

## Opt-out

`--no-hwaccel` on the `oxideav` CLI biases dispatch away from HW factories without unregistering them.

## Coverage roadmap

| Codec        | Decode | Encode |
|--------------|--------|--------|
| H.264        | planned | planned |
| HEVC         | planned | planned |
| AV1          | planned (vendor support varies) | planned |
| VP9          | — | — |

Round 1 (this commit): scaffolding only. The crate dlopens `libvulkan.so.1`, resolves the four bootstrap symbols listed above, and exposes a `register(&mut RuntimeContext)` entry point that confirms the loader loads without registering any codec factories yet. Round 2: instance + physical-device enumeration + extension probing + first decode codec.

## Workspace policy

Calling a system OS / driver API via FFI is the same shape as calling `libc::malloc` — it's the platform, not a copied algorithm. The workspace's clean-room rule (no embedding source from libvpx, libwebp, libjxl, etc.) does not apply to this crate.

## License

MIT.
