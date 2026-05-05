# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial scaffolding: `#![cfg(target_os = "linux")]` crate that
  dlopens `libvulkan.so.1` via `libloading` on first use.
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
  symbol resolution on Linux machines that have a Vulkan loader
  installed.
