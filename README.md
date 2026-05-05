# oxideav-vulkan-video

Vulkan Video hardware decode/encode bridge for the [oxideav](https://github.com/OxideAV/oxideav) framework. Builds on **Linux and Windows**.

## Why a bridge crate?

The Vulkan Video extension family (`VK_KHR_video_queue`, `VK_KHR_video_decode_h264`, `VK_KHR_video_decode_h265`, `VK_KHR_video_decode_av1`, `VK_KHR_video_encode_*`) is the **vendor- and OS-neutral** path forward for HW acceleration. Unlike VA-API (Intel/AMD-leaning, Linux-only) and NVENC (NVIDIA-only), Vulkan Video is implemented in the Vulkan ICD layer itself and is gradually shipping across all three major GPU vendors on both Linux and Windows. As of 2025, decode is widely available; encode is rolling out.

This crate is a **thin runtime-loaded bridge** — no compile-time link dependency on the Vulkan loader or any vendor ICD. The loader is opened via [`libloading`] on first use:

| Platform | Loader filename |
|----------|-----------------|
| Linux    | `libvulkan.so.1` |
| Windows  | `vulkan-1.dll`   |

On Windows the loader is installed by the Vulkan SDK, by GPU driver packages (NVIDIA, AMD, Intel), and by Windows itself on recent builds.

## Programming model

Vulkan is unusual in that **only `vkGetInstanceProcAddr` is meaningfully resolved by `dlsym`**. Every other Vulkan function — including all video extension entry points (`vkCmdBeginVideoCodingKHR`, `vkGetVideoSessionMemoryRequirementsKHR`, …) — is reached via `vkGetInstanceProcAddr` (instance-level entries) or `vkGetDeviceProcAddr` (device-level entries) after a `VkInstance` is created. So this crate's bootstrap vtable is intentionally tiny:

* `vkGetInstanceProcAddr`
* `vkCreateInstance`
* `vkEnumerateInstanceExtensionProperties`
* `vkEnumerateInstanceVersion`

As of Round 2, the crate uses these to construct a `VkInstance`, enumerate physical devices, and probe the `VK_KHR_video_*` extension family. Every other Vulkan entry is resolved on demand through `vkGetInstanceProcAddr` / `vkGetDeviceProcAddr`.

## Fallback behaviour

Two distinct failure paths fall back automatically to the pure-Rust codec:

1. **Load failure** — Vulkan loader not installed, no Vulkan ICD on the system (e.g. headless Linux CI without Mesa, Windows host without GPU driver). `register()` logs and returns without registering.
2. **Init failure** — `vkCreateInstance` succeeds but `vkEnumerateDeviceExtensionProperties` reports the requested `VK_KHR_video_*` extension is unsupported by every available `VkPhysicalDevice`, or the queue family for video-decode/encode operations is missing. The factory returns `Err`; the registry falls back to the next-priority impl.

Pipelines that **require** hardware can opt out of the SW fallback by setting `CodecPreferences { require_hardware: true, .. }`.

## Platform gating

The whole crate is `#![cfg(any(target_os = "linux", target_os = "windows"))]`. On macOS it compiles to an empty rlib; the umbrella `oxideav` crate gates the `register` call behind the same cfg. (Vulkan is reachable on macOS via MoltenVK but with a different loading story — out of scope for now.)

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

Round 2 (this commit): the bootstrap → `VkInstance` → `VkPhysicalDevice` → `VK_KHR_video_*` extension probe path is plumbed end-to-end. `Instance::new("oxideav-vulkan-video-test", VK_API_VERSION_1_2)` calls `vkCreateInstance`, resolves the post-bootstrap function pointers through `vkGetInstanceProcAddr`, and exposes safe wrappers for `vkEnumeratePhysicalDevices`, `vkGetPhysicalDeviceProperties`, `vkEnumerateDeviceExtensionProperties`, and `vkGetPhysicalDeviceQueueFamilyProperties2` (the `_2` form gives a `pNext` chain into `VkQueueFamilyVideoPropertiesKHR`). `PhysicalDevice::supports_video_extensions()` returns a per-codec bool summary (queue_khr, decode_h264, decode_h265, decode_av1, encode_h264, encode_h265). Verified on an NVIDIA RTX 5080 with driver 580.95.05 in `tests/round2_init.rs`: every codec extension above is advertised and 2 queue families carry `VK_QUEUE_VIDEO_DECODE_BIT_KHR` / `VK_QUEUE_VIDEO_ENCODE_BIT_KHR`. `register()` remains a graceful no-op — Round 3 will layer the first decode session (H.264 / HEVC) on top.

## Workspace policy

Calling a system OS / driver API via FFI is the same shape as calling `libc::malloc` — it's the platform, not a copied algorithm. The workspace's clean-room rule (no embedding source from libvpx, libwebp, libjxl, etc.) does not apply to this crate.

## License

MIT.
