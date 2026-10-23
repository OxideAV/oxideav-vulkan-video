//! Safe wrapper over `VkDevice` (logical device).
//!
//! Constructed via [`Device::new`] from a parent
//! [`PhysicalDevice`]. Holds the device handle plus the resolved
//! device-level dispatch table (every function reachable through
//! `vkGetDeviceProcAddr`). Drop calls `vkDestroyDevice`.
//!
//! Round 3 wires up enough of the device dispatch surface to:
//!
//! * fetch a `VkQueue` from a video-decode queue family
//!   (`Device::queue`),
//! * resolve `vkAllocateMemory` / `vkFreeMemory` so video session
//!   memory binding works,
//! * resolve `vkCreateVideoSessionKHR` /
//!   `vkDestroyVideoSessionKHR` /
//!   `vkGetVideoSessionMemoryRequirementsKHR` /
//!   `vkBindVideoSessionMemoryKHR` so [`crate::video::VideoSession`]
//!   can call them.
//!
//! Logical devices, like instances, are externally synchronised in
//! Vulkan; we don't mark `Device` as `Send + Sync` and leave thread
//! affinity to the caller.

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr;

use crate::instance::{VkError, load_device_fn};
use crate::physical_device::PhysicalDevice;
use crate::sys::{
    FnVkAllocateCommandBuffers, FnVkAllocateMemory, FnVkBeginCommandBuffer, FnVkBindBufferMemory,
    FnVkBindImageMemory, FnVkBindVideoSessionMemoryKHR, FnVkCmdBeginVideoCodingKHR,
    FnVkCmdControlVideoCodingKHR, FnVkCmdCopyImageToBuffer, FnVkCmdDecodeVideoKHR,
    FnVkCmdEndVideoCodingKHR, FnVkCmdPipelineBarrier, FnVkCreateBuffer, FnVkCreateCommandPool,
    FnVkCreateFence, FnVkCreateImage, FnVkCreateImageView, FnVkCreateVideoSessionKHR,
    FnVkCreateVideoSessionParametersKHR, FnVkDestroyBuffer, FnVkDestroyCommandPool,
    FnVkDestroyDevice, FnVkDestroyFence, FnVkDestroyImage, FnVkDestroyImageView,
    FnVkDestroyVideoSessionKHR, FnVkDestroyVideoSessionParametersKHR, FnVkEndCommandBuffer,
    FnVkFreeCommandBuffers, FnVkFreeMemory, FnVkGetBufferMemoryRequirements, FnVkGetDeviceProcAddr,
    FnVkGetDeviceQueue, FnVkGetImageMemoryRequirements, FnVkGetVideoSessionMemoryRequirementsKHR,
    FnVkMapMemory, FnVkQueueSubmit, FnVkQueueWaitIdle, FnVkUnmapMemory, FnVkWaitForFences,
    VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, VK_SUCCESS,
    VkDevice, VkDeviceCreateInfo, VkDeviceQueueCreateInfo, VkQueue,
};

/// Default per-queue priority. Vulkan only requires the values to lie
/// in `[0.0, 1.0]`; the relative weight matters only when multiple
/// queue families compete for execution units.
const DEFAULT_QUEUE_PRIORITY: f32 = 1.0;

/// A live `VkDevice` plus the post-create dispatch surface.
///
/// `Device` owns the handle: `Drop` calls `vkDestroyDevice`. The
/// borrow checker prevents constructing one without holding the
/// parent [`PhysicalDevice`] (and through it the [`crate::Instance`])
/// alive — the spec rule that the device must not outlive its
/// instance is enforced by the lifetime tied to the
/// `&InstanceFns` reference held internally.
pub struct Device {
    handle: VkDevice,
    fns: DeviceFns,
    /// Whether `Drop` should call `vkDestroyDevice`. `true` for
    /// devices built by [`Device::new`]; `false` for foreign handles
    /// imported via [`Device::from_raw`] — those are owned by the
    /// caller and must be destroyed by the caller.
    owned: bool,
}

/// Function pointers resolved via `vkGetDeviceProcAddr` against a
/// freshly created `VkDevice`. Per Vulkan, device-dispatch is always
/// preferred over instance-dispatch for entry points that operate on
/// or below a `VkDevice` — the loader can avoid an extra layer of
/// trampolines this way.
#[allow(dead_code)]
pub(crate) struct DeviceFns {
    pub(crate) get_device_proc_addr: FnVkGetDeviceProcAddr,
    pub(crate) destroy_device: FnVkDestroyDevice,
    pub(crate) get_device_queue: FnVkGetDeviceQueue,
    pub(crate) allocate_memory: FnVkAllocateMemory,
    pub(crate) free_memory: FnVkFreeMemory,
    pub(crate) map_memory: FnVkMapMemory,
    pub(crate) unmap_memory: FnVkUnmapMemory,
    pub(crate) create_video_session_khr: FnVkCreateVideoSessionKHR,
    pub(crate) destroy_video_session_khr: FnVkDestroyVideoSessionKHR,
    pub(crate) get_video_session_memory_requirements_khr: FnVkGetVideoSessionMemoryRequirementsKHR,
    pub(crate) bind_video_session_memory_khr: FnVkBindVideoSessionMemoryKHR,

    // Round 4 additions
    pub(crate) create_video_session_parameters_khr: FnVkCreateVideoSessionParametersKHR,
    pub(crate) destroy_video_session_parameters_khr: FnVkDestroyVideoSessionParametersKHR,
    pub(crate) cmd_begin_video_coding_khr: FnVkCmdBeginVideoCodingKHR,
    pub(crate) cmd_end_video_coding_khr: FnVkCmdEndVideoCodingKHR,
    pub(crate) cmd_control_video_coding_khr: FnVkCmdControlVideoCodingKHR,
    pub(crate) cmd_decode_video_khr: FnVkCmdDecodeVideoKHR,

    pub(crate) create_buffer: FnVkCreateBuffer,
    pub(crate) destroy_buffer: FnVkDestroyBuffer,
    pub(crate) create_image: FnVkCreateImage,
    pub(crate) destroy_image: FnVkDestroyImage,
    pub(crate) create_image_view: FnVkCreateImageView,
    pub(crate) destroy_image_view: FnVkDestroyImageView,
    pub(crate) get_buffer_memory_requirements: FnVkGetBufferMemoryRequirements,
    pub(crate) get_image_memory_requirements: FnVkGetImageMemoryRequirements,
    pub(crate) bind_buffer_memory: FnVkBindBufferMemory,
    pub(crate) bind_image_memory: FnVkBindImageMemory,

    pub(crate) create_command_pool: FnVkCreateCommandPool,
    pub(crate) destroy_command_pool: FnVkDestroyCommandPool,
    pub(crate) allocate_command_buffers: FnVkAllocateCommandBuffers,
    pub(crate) free_command_buffers: FnVkFreeCommandBuffers,
    pub(crate) begin_command_buffer: FnVkBeginCommandBuffer,
    pub(crate) end_command_buffer: FnVkEndCommandBuffer,
    pub(crate) cmd_pipeline_barrier: FnVkCmdPipelineBarrier,
    pub(crate) cmd_copy_image_to_buffer: FnVkCmdCopyImageToBuffer,

    pub(crate) queue_submit: FnVkQueueSubmit,
    pub(crate) queue_wait_idle: FnVkQueueWaitIdle,

    pub(crate) create_fence: FnVkCreateFence,
    pub(crate) destroy_fence: FnVkDestroyFence,
    pub(crate) wait_for_fences: FnVkWaitForFences,
}

/// A `VkQueue` handle. Round 3 doesn't issue commands yet — the
/// handle is just the entry point a future round will use to submit
/// the decode command buffer.
#[derive(Copy, Clone)]
pub struct Queue {
    pub(crate) handle: VkQueue,
    pub(crate) family_index: u32,
}

impl Queue {
    /// Raw `VkQueue` handle. For interop / tests.
    pub fn handle(&self) -> VkQueue {
        self.handle
    }

    /// Queue-family index this queue was retrieved from.
    pub fn family_index(&self) -> u32 {
        self.family_index
    }
}

impl Device {
    /// Construct a logical device with a single queue from
    /// `video_decode_queue_family_index` and the supplied list of
    /// extension names enabled.
    ///
    /// Caller is responsible for picking a queue family that
    /// advertises `VK_QUEUE_VIDEO_DECODE_BIT_KHR` (use
    /// [`PhysicalDevice::video_queue_family_indices`]) and for
    /// passing the matching extension names — at minimum
    /// `VK_KHR_video_queue`, `VK_KHR_video_decode_queue`, and the
    /// codec-specific decode extension (e.g.
    /// `VK_KHR_video_decode_h264`).
    ///
    /// Returns `VkError::Result` if `vkCreateDevice` fails (most
    /// commonly because one of the requested extensions is not
    /// supported by this physical device).
    pub fn new(
        physical_device: &PhysicalDevice<'_>,
        video_decode_queue_family_index: u32,
        requested_extensions: &[&str],
    ) -> Result<Self, VkError> {
        let fns = physical_device.instance_fns();

        // Per-queue priority is held by reference from
        // VkDeviceQueueCreateInfo, so it has to outlive the call.
        let priorities = [DEFAULT_QUEUE_PRIORITY];

        let queue_ci = VkDeviceQueueCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            queue_family_index: video_decode_queue_family_index,
            queue_count: 1,
            p_queue_priorities: priorities.as_ptr(),
        };

        // Convert the requested extension names into a contiguous
        // pointer table that lives until after `vkCreateDevice`
        // returns. The CStrings own the storage; the pointer vector
        // holds borrows into them.
        let ext_cstrings: Vec<CString> = requested_extensions
            .iter()
            .map(|s| CString::new(*s).unwrap_or_else(|_| CString::new("").unwrap()))
            .collect();
        let ext_ptrs: Vec<*const c_char> = ext_cstrings.iter().map(|s| s.as_ptr()).collect();

        let create_info = VkDeviceCreateInfo {
            s_type: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: ptr::null(),
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_ci,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: ext_ptrs.len() as u32,
            pp_enabled_extension_names: if ext_ptrs.is_empty() {
                ptr::null()
            } else {
                ext_ptrs.as_ptr()
            },
            p_enabled_features: ptr::null(),
        };

        let mut device: VkDevice = ptr::null_mut();
        // SAFETY: `create_info` references `queue_ci`, `priorities`,
        // and the `ext_*` allocations, all of which live until after
        // this call returns. Vulkan must copy any data it needs to
        // retain before returning per the spec.
        let result = unsafe {
            (fns.create_device)(
                physical_device.handle(),
                &create_info,
                ptr::null(),
                &mut device,
            )
        };
        if result != VK_SUCCESS {
            return Err(VkError::Result {
                op: "vkCreateDevice",
                result,
            });
        }

        let device_fns = DeviceFns::resolve(fns.get_device_proc_addr, device)?;

        Ok(Self {
            handle: device,
            fns: device_fns,
            owned: true,
        })
    }

    /// Import a foreign `VkDevice` created by the application (e.g.
    /// the same logical device a renderer built through `ash` /
    /// `vulkano` / `wgpu`-hal) instead of creating a new one.
    ///
    /// The returned wrapper is **non-owning**: `Drop` does *not* call
    /// `vkDestroyDevice` — the caller keeps ownership and destroys
    /// the device after every object derived from this wrapper is
    /// gone. The device-level dispatch table is resolved through the
    /// parent instance's `vkGetDeviceProcAddr`, so `physical_device`
    /// must come from the same [`crate::Instance`] wrapper the device
    /// was created against (for imported instances, see
    /// [`crate::Instance::from_raw`] and
    /// [`crate::Instance::physical_device_from_raw`]).
    ///
    /// Returns `VkError::MissingFunction` if a required device entry
    /// point can't be resolved — most commonly because the device was
    /// created **without** the video extensions enabled. For H.264
    /// decode the device needs at least `VK_KHR_video_queue`,
    /// `VK_KHR_video_decode_queue`, `VK_KHR_video_decode_h264`, and
    /// (below Vulkan 1.3) `VK_KHR_synchronization2`.
    ///
    /// # Safety
    ///
    /// * `handle` must be a valid, live `VkDevice` created from
    ///   `physical_device`'s underlying `VkPhysicalDevice`, against
    ///   the same instance `physical_device` borrows from.
    /// * The device must outlive the returned wrapper and every
    ///   object derived from it (video sessions, images, buffers,
    ///   command pools, …).
    /// * Logical devices are externally synchronised: the caller must
    ///   not destroy the device — or submit to any queue this wrapper
    ///   uses — concurrently with calls through the wrapper.
    pub unsafe fn from_raw(
        physical_device: &PhysicalDevice<'_>,
        handle: VkDevice,
    ) -> Result<Self, VkError> {
        let fns = physical_device.instance_fns();
        let device_fns = DeviceFns::resolve(fns.get_device_proc_addr, handle)?;
        Ok(Self {
            handle,
            fns: device_fns,
            owned: false,
        })
    }

    /// Raw `VkDevice` handle. Most callers should not need this;
    /// expand the safe wrappers instead.
    pub fn handle(&self) -> VkDevice {
        self.handle
    }

    /// Crate-internal accessor for the device dispatch table — used
    /// by [`crate::video`] to issue session calls without exposing
    /// the table publicly.
    pub(crate) fn fns(&self) -> &DeviceFns {
        &self.fns
    }

    /// Retrieve queue 0 from `family_index`. Round 3 only ever
    /// requests one queue per family in [`Device::new`].
    pub fn queue(&self, family_index: u32) -> Queue {
        self.queue_indexed(family_index, 0)
    }

    /// Retrieve queue `queue_index` from `family_index`. Useful for
    /// imported devices ([`Device::from_raw`]) whose owner created
    /// more than one queue in the family and reserves index 0 for
    /// its own (e.g. graphics) submissions.
    pub fn queue_indexed(&self, family_index: u32, queue_index: u32) -> Queue {
        let mut q: VkQueue = ptr::null_mut();
        // SAFETY: `device` is a valid handle and `(family_index,
        // queue_index)` is assumed to match what was passed to
        // `vkCreateDevice`. The call cannot fail under those
        // preconditions.
        unsafe { (self.fns.get_device_queue)(self.handle, family_index, queue_index, &mut q) }
        Queue {
            handle: q,
            family_index,
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if self.owned && !self.handle.is_null() {
            // SAFETY: the spec requires every child object (queues,
            // sessions, memory, …) to be destroyed before the device.
            // Outstanding child wrappers in this crate borrow `&Device`
            // so the borrow checker prevents calling drop while one is
            // live.
            unsafe {
                (self.fns.destroy_device)(self.handle, ptr::null());
            }
            self.handle = ptr::null_mut();
        }
    }
}

impl DeviceFns {
    fn resolve(get_device_proc: FnVkGetDeviceProcAddr, device: VkDevice) -> Result<Self, VkError> {
        // SAFETY: `vkGetDeviceProcAddr` has the spec-declared
        // signature for each name; null is "not present".
        unsafe {
            Ok(Self {
                get_device_proc_addr: get_device_proc,
                destroy_device: load_device_fn(get_device_proc, device, b"vkDestroyDevice\0")?,
                get_device_queue: load_device_fn(get_device_proc, device, b"vkGetDeviceQueue\0")?,
                allocate_memory: load_device_fn(get_device_proc, device, b"vkAllocateMemory\0")?,
                free_memory: load_device_fn(get_device_proc, device, b"vkFreeMemory\0")?,
                map_memory: load_device_fn(get_device_proc, device, b"vkMapMemory\0")?,
                unmap_memory: load_device_fn(get_device_proc, device, b"vkUnmapMemory\0")?,
                create_video_session_khr: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkCreateVideoSessionKHR\0",
                )?,
                destroy_video_session_khr: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkDestroyVideoSessionKHR\0",
                )?,
                get_video_session_memory_requirements_khr: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkGetVideoSessionMemoryRequirementsKHR\0",
                )?,
                bind_video_session_memory_khr: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkBindVideoSessionMemoryKHR\0",
                )?,
                create_video_session_parameters_khr: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkCreateVideoSessionParametersKHR\0",
                )?,
                destroy_video_session_parameters_khr: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkDestroyVideoSessionParametersKHR\0",
                )?,
                cmd_begin_video_coding_khr: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkCmdBeginVideoCodingKHR\0",
                )?,
                cmd_end_video_coding_khr: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkCmdEndVideoCodingKHR\0",
                )?,
                cmd_control_video_coding_khr: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkCmdControlVideoCodingKHR\0",
                )?,
                cmd_decode_video_khr: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkCmdDecodeVideoKHR\0",
                )?,
                create_buffer: load_device_fn(get_device_proc, device, b"vkCreateBuffer\0")?,
                destroy_buffer: load_device_fn(get_device_proc, device, b"vkDestroyBuffer\0")?,
                create_image: load_device_fn(get_device_proc, device, b"vkCreateImage\0")?,
                destroy_image: load_device_fn(get_device_proc, device, b"vkDestroyImage\0")?,
                create_image_view: load_device_fn(get_device_proc, device, b"vkCreateImageView\0")?,
                destroy_image_view: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkDestroyImageView\0",
                )?,
                get_buffer_memory_requirements: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkGetBufferMemoryRequirements\0",
                )?,
                get_image_memory_requirements: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkGetImageMemoryRequirements\0",
                )?,
                bind_buffer_memory: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkBindBufferMemory\0",
                )?,
                bind_image_memory: load_device_fn(get_device_proc, device, b"vkBindImageMemory\0")?,
                create_command_pool: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkCreateCommandPool\0",
                )?,
                destroy_command_pool: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkDestroyCommandPool\0",
                )?,
                allocate_command_buffers: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkAllocateCommandBuffers\0",
                )?,
                free_command_buffers: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkFreeCommandBuffers\0",
                )?,
                begin_command_buffer: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkBeginCommandBuffer\0",
                )?,
                end_command_buffer: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkEndCommandBuffer\0",
                )?,
                cmd_pipeline_barrier: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkCmdPipelineBarrier\0",
                )?,
                cmd_copy_image_to_buffer: load_device_fn(
                    get_device_proc,
                    device,
                    b"vkCmdCopyImageToBuffer\0",
                )?,
                queue_submit: load_device_fn(get_device_proc, device, b"vkQueueSubmit\0")?,
                queue_wait_idle: load_device_fn(get_device_proc, device, b"vkQueueWaitIdle\0")?,
                create_fence: load_device_fn(get_device_proc, device, b"vkCreateFence\0")?,
                destroy_fence: load_device_fn(get_device_proc, device, b"vkDestroyFence\0")?,
                wait_for_fences: load_device_fn(get_device_proc, device, b"vkWaitForFences\0")?,
            })
        }
    }
}

// ─────────────────────────── ExternalDevice ──────────────────────────────────

/// Raw handles describing an application-owned Vulkan device to run
/// video decode on, instead of letting the crate create its own
/// instance / device (GitHub issue #2).
///
/// Pass this to `H264VkDecoder::make_with_device` (with the
/// default-on `registry` feature), or use the lower-level
/// [`crate::Instance::from_raw`] / [`Device::from_raw`] wrappers
/// directly.
///
/// # Requirements on the imported objects
///
/// * The instance must be Vulkan 1.1+ (the crate calls
///   `vkGetPhysicalDeviceQueueFamilyProperties2`, core in 1.1).
/// * The device must have been created with the video extensions
///   enabled: `VK_KHR_video_queue`, `VK_KHR_video_decode_queue`, the
///   codec extension (`VK_KHR_video_decode_h264` for the H.264
///   decoder), and — below Vulkan 1.3 — `VK_KHR_synchronization2`.
/// * `queue_family_index` must name a queue family that advertises
///   `VK_QUEUE_VIDEO_DECODE_BIT_KHR`, and the device must have been
///   created with at least `queue_index + 1` queues in that family.
///
/// # Ownership & synchronisation
///
/// The application keeps ownership of all three handles; nothing in
/// this struct is destroyed by the crate. The handles must stay valid
/// for the whole lifetime of every wrapper / decoder constructed from
/// them. The decoder submits to — and waits idle on — the named
/// queue, so that queue must not be used concurrently from other
/// threads while a decode call is in flight.
#[derive(Clone, Copy)]
pub struct ExternalDevice {
    /// The application's `VkInstance`.
    pub instance: crate::sys::VkInstance,
    /// The `VkPhysicalDevice` the logical device was created from.
    pub physical_device: crate::sys::VkPhysicalDevice,
    /// The application's `VkDevice`.
    pub device: VkDevice,
    /// Index of a video-decode-capable queue family the device was
    /// created with.
    pub queue_family_index: u32,
    /// Which queue within `queue_family_index` the decoder should
    /// submit to. `0` unless the application reserves queue 0 of the
    /// family for its own submissions.
    pub queue_index: u32,
    /// `vkGetInstanceProcAddr` of the loader the instance was created
    /// with. `None` means "resolve through the system Vulkan loader"
    /// (correct whenever the application itself went through
    /// `libvulkan.so.1` / `vulkan-1.dll`, which is the overwhelmingly
    /// common case — including `ash::Entry::load()`).
    pub get_instance_proc_addr: Option<crate::sys::FnVkGetInstanceProcAddr>,
}

impl std::fmt::Debug for ExternalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalDevice")
            .field("instance", &self.instance)
            .field("physical_device", &self.physical_device)
            .field("device", &self.device)
            .field("queue_family_index", &self.queue_family_index)
            .field("queue_index", &self.queue_index)
            .field(
                "get_instance_proc_addr",
                &self.get_instance_proc_addr.map(|_| "<fn>"),
            )
            .finish()
    }
}

impl ExternalDevice {
    /// Convenience constructor with `queue_index = 0` and the system
    /// loader for function resolution.
    pub fn new(
        instance: crate::sys::VkInstance,
        physical_device: crate::sys::VkPhysicalDevice,
        device: VkDevice,
        queue_family_index: u32,
    ) -> Self {
        Self {
            instance,
            physical_device,
            device,
            queue_family_index,
            queue_index: 0,
            get_instance_proc_addr: None,
        }
    }

    /// Builder-style override of [`ExternalDevice::queue_index`].
    pub fn with_queue_index(mut self, queue_index: u32) -> Self {
        self.queue_index = queue_index;
        self
    }

    /// Builder-style override of
    /// [`ExternalDevice::get_instance_proc_addr`] for applications
    /// using a non-standard loader.
    pub fn with_get_instance_proc_addr(
        mut self,
        get_instance_proc_addr: crate::sys::FnVkGetInstanceProcAddr,
    ) -> Self {
        self.get_instance_proc_addr = Some(get_instance_proc_addr);
        self
    }

    /// Import the handles as non-owning wrapper objects:
    /// `(Instance, VkPhysicalDevice handle)` — the second element is
    /// re-wrapped per use via
    /// [`crate::Instance::physical_device_from_raw`] because
    /// `PhysicalDevice` borrows the returned `Instance`.
    ///
    /// # Safety
    ///
    /// Same contract as [`crate::Instance::from_raw`] and
    /// [`Device::from_raw`]: all three handles are valid, mutually
    /// consistent (device from physical device from instance), stay
    /// alive for the lifetime of the returned wrappers, and are not
    /// destroyed / used concurrently while the wrappers are in use.
    pub unsafe fn import(&self) -> Result<(crate::Instance, Device), VkError> {
        // SAFETY: forwarded caller contract.
        let instance = unsafe {
            match self.get_instance_proc_addr {
                Some(gipa) => crate::Instance::from_raw_with_loader(self.instance, gipa)?,
                None => crate::Instance::from_raw(self.instance)?,
            }
        };
        // SAFETY: `physical_device` belongs to `instance` per the
        // caller contract; the returned `PhysicalDevice` only lives
        // for this call.
        let device = unsafe {
            let pd = instance.physical_device_from_raw(self.physical_device);
            Device::from_raw(&pd, self.device)?
        };
        Ok((instance, device))
    }
}
