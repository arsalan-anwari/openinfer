use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::Result;
use ash::vk;

mod buffers;
mod device;
mod dispatch;
mod pipeline;
mod spirv;

pub use device::VulkanCaps;

#[allow(dead_code)]
pub struct VulkanRuntime {
    caps: VulkanCaps,
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    query_pool: Option<vk::QueryPool>,
    timestamp_period: f32,
    last_dispatch_ns: Mutex<Option<u64>>,
}

impl VulkanRuntime {
    pub fn caps(&self) -> VulkanCaps {
        self.caps
    }

    pub fn take_last_dispatch_duration(&self) -> Option<Duration> {
        let mut guard = self.last_dispatch_ns.lock().ok()?;
        guard.take().map(Duration::from_nanos)
    }

}

static VULKAN_RUNTIME: OnceLock<VulkanRuntime> = OnceLock::new();

pub fn set_vulkan_runtime(runtime: VulkanRuntime) -> Result<()> {
    if VULKAN_RUNTIME.set(runtime).is_err() {
        // Another thread already initialized the runtime.
        return Ok(());
    }
    Ok(())
}

pub fn get_vulkan_runtime() -> Option<&'static VulkanRuntime> {
    VULKAN_RUNTIME.get()
}

pub fn take_last_dispatch_duration() -> Option<Duration> {
    get_vulkan_runtime()?.take_last_dispatch_duration()
}
