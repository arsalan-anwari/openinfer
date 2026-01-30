use std::sync::Mutex;

use anyhow::{anyhow, Result};
use ash::vk;

use crate::tensor::DType;
use crate::vk_trace;

use super::VulkanRuntime;

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct VulkanCaps {
    pub int64: bool,
    pub float64: bool,
    pub subgroup: bool,
}

impl VulkanCaps {
    pub fn supports_dtype(&self, dtype: DType) -> bool {
        match dtype {
            DType::I64 | DType::U64 => self.int64,
            DType::F64 => self.float64,
            _ => true,
        }
    }
}

#[allow(dead_code)]
impl VulkanRuntime {
    pub fn new(_caps: VulkanCaps) -> Result<Self> {
        vk_trace!("initializing Vulkan runtime");
        let entry = unsafe { ash::Entry::load()? };
        let app_name = b"openinfer\0";
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr() as *const i8,
            application_version: 0,
            p_engine_name: app_name.as_ptr() as *const i8,
            engine_version: 0,
            api_version: vk::make_api_version(0, 1, 1, 0),
            ..Default::default()
        };
        let instance_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };
        let instance = unsafe { entry.create_instance(&instance_info, None)? };
        vk_trace!("created Vulkan instance");
        let physical_device = unsafe {
            instance
                .enumerate_physical_devices()?
                .get(0)
                .copied()
                .ok_or_else(|| anyhow!("no Vulkan physical devices found"))?
        };
        vk_trace!("selected physical device {:?}", physical_device);
        let device_features = unsafe { instance.get_physical_device_features(physical_device) };
        let caps = VulkanCaps {
            int64: device_features.shader_int64 == vk::TRUE,
            float64: device_features.shader_float64 == vk::TRUE,
            subgroup: false,
        };
        vk_trace!(
            "vulkan caps: int64={} float64={}",
            caps.int64,
            caps.float64
        );
        let queue_family_props = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };
        let queue_family_index = queue_family_props
            .iter()
            .enumerate()
            .find(|(_, props)| {
                props.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && props.timestamp_valid_bits > 0
            })
            .or_else(|| {
                queue_family_props
                    .iter()
                    .enumerate()
                    .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
            })
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| anyhow!("no Vulkan compute queue family found"))?;
        let timestamp_supported = queue_family_props
            .get(queue_family_index as usize)
            .map(|props| props.timestamp_valid_bits > 0)
            .unwrap_or(false);
        let physical_props = unsafe { instance.get_physical_device_properties(physical_device) };
        let timestamp_period = physical_props.limits.timestamp_period;
        vk_trace!("using compute queue family {}", queue_family_index);
        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo {
            queue_family_index,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };
        let device_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            p_enabled_features: &device_features,
            ..Default::default()
        };
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        vk_trace!("created logical device and queue");
        let command_pool_info = vk::CommandPoolCreateInfo {
            queue_family_index,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };
        let query_pool = if timestamp_supported {
            let query_info = vk::QueryPoolCreateInfo {
                query_type: vk::QueryType::TIMESTAMP,
                query_count: 2,
                ..Default::default()
            };
            Some(unsafe { device.create_query_pool(&query_info, None)? })
        } else {
            None
        };
        if timestamp_supported {
            vk_trace!("timestamp queries supported (period={}ns)", timestamp_period);
        }

        Ok(Self {
            caps,
            entry,
            instance,
            device,
            physical_device,
            queue,
            queue_family_index,
            command_pool,
            query_pool,
            timestamp_period,
            last_dispatch_ns: Mutex::new(None),
        })
    }
}
