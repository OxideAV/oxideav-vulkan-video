# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
