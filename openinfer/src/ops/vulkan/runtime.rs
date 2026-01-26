use std::fs;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, Result};
use ash::vk;

use crate::vk_trace;
use crate::ops::vulkan::spv::embedded_spv;

use crate::tensor::DType;

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct VulkanCaps {
    pub int64: bool,
    pub float64: bool,
    pub float16: bool,
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
pub struct VulkanRuntime {
    caps: VulkanCaps,
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue: vk::Queue,
    queue_family_index: u32,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    query_pool: Option<vk::QueryPool>,
    timestamp_period: f32,
    last_dispatch_ns: Mutex<Option<u64>>,
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
        let mut float16_int8 = vk::PhysicalDeviceFloat16Int8FeaturesKHR {
            ..Default::default()
        };
        let mut features2 = vk::PhysicalDeviceFeatures2 {
            ..Default::default()
        };
        unsafe {
            features2.p_next = &mut float16_int8 as *mut _ as *mut std::ffi::c_void;
            instance.get_physical_device_features2(physical_device, &mut features2);
        }
        let caps = VulkanCaps {
            int64: device_features.shader_int64 == vk::TRUE,
            float64: device_features.shader_float64 == vk::TRUE,
            float16: float16_int8.shader_float16 == vk::TRUE,
            subgroup: false,
        };
        vk_trace!(
            "vulkan caps: int64={} float64={} float16={}",
            caps.int64,
            caps.float64,
            caps.float16
        );
        let queue_family_props = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
        };
        let queue_family_index = queue_family_props
            .iter()
            .enumerate()
            .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::COMPUTE))
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
        let mut enabled_float16_int8 = vk::PhysicalDeviceFloat16Int8FeaturesKHR {
            shader_float16: if caps.float16 { vk::TRUE } else { vk::FALSE },
            ..Default::default()
        };
        let device_info = vk::DeviceCreateInfo {
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_info,
            p_enabled_features: &device_features,
            p_next: &mut enabled_float16_int8 as *mut _ as *mut std::ffi::c_void,
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
        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&set_layout_info, None)? };
        vk_trace!("created descriptor set layout");
        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: 16,
        };
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_range,
            ..Default::default()
        };
        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };
        vk_trace!("created pipeline layout with push constants");
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
        vk_trace!("created descriptor pool");
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
            descriptor_pool,
            descriptor_set_layout,
            pipeline_layout,
            query_pool,
            timestamp_period,
            last_dispatch_ns: Mutex::new(None),
        })
    }

    pub fn caps(&self) -> VulkanCaps {
        self.caps
    }

    pub fn take_last_dispatch_duration(&self) -> Option<Duration> {
        let mut guard = self.last_dispatch_ns.lock().ok()?;
        guard.take().map(Duration::from_nanos)
    }

    pub fn dispatch_add(
        &self,
        target: &str,
        tensor_descs: &[u8],
        input_bytes: &[u8],
        output_bytes: &mut [u8],
        push_constants: &[u8],
        output_offset: u64,
        output_alias_input: bool,
    ) -> Result<()> {
        if let Ok(mut guard) = self.last_dispatch_ns.lock() {
            *guard = None;
        }
        let spv_path = format!("src/ops/vulkan/add/bin/{}.spv", target);
        vk_trace!(
            "dispatch_add target={} desc_bytes={} input_bytes={} output_bytes={}",
            target,
            tensor_descs.len(),
            input_bytes.len(),
            output_bytes.len()
        );
        let spv_bytes = if let Some(bytes) = embedded_spv(target) {
            bytes.to_vec()
        } else {
            let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&spv_path);
            fs::read(&spv_path)
                .or_else(|_| fs::read(&manifest_path))
                .map_err(|err| anyhow!("failed to read {}: {}", spv_path, err))?
        };
        validate_spirv(&spv_bytes, target)?;
        let spv_u32 = bytemuck::cast_slice::<u8, u32>(&spv_bytes);
        let entrypoints = spirv_entrypoints(spv_u32);
        if !entrypoints.is_empty() {
            vk_trace!("spv entrypoints: {:?}", entrypoints);
        }
        let shader_info = vk::ShaderModuleCreateInfo {
            code_size: spv_bytes.len(),
            p_code: spv_u32.as_ptr(),
            ..Default::default()
        };
        let shader_module = unsafe { self.device.create_shader_module(&shader_info, None)? };
        let entry_name = select_entrypoint(&entrypoints, target);
        let entry_name = std::ffi::CString::new(entry_name)?;
        let stage_info = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader_module,
            p_name: entry_name.as_ptr(),
            ..Default::default()
        };
        let pipeline_info = vk::ComputePipelineCreateInfo {
            stage: stage_info,
            layout: self.pipeline_layout,
            ..Default::default()
        };
        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_info),
                    None,
                )
                .map_err(|err| anyhow!("failed to create compute pipeline: {:?}", err))?
        }[0];
        vk_trace!("created compute pipeline");

        unsafe { self.device.reset_descriptor_pool(self.descriptor_pool, vk::DescriptorPoolResetFlags::empty())? };
        let set_alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &self.descriptor_set_layout,
            ..Default::default()
        };
        let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&set_alloc_info)?[0] };
        vk_trace!("allocated descriptor set");

        let desc_buffer = self.create_buffer(tensor_descs.len() as u64)?;
        let input_buffer = self.create_buffer(input_bytes.len() as u64)?;
        let output_buffer = if output_alias_input {
            None
        } else {
            Some(self.create_buffer(output_bytes.len() as u64)?)
        };
        vk_trace!("created buffers (desc/input/output)");

        unsafe {
            self.write_buffer(desc_buffer, tensor_descs)?;
            self.write_buffer(input_buffer, input_bytes)?;
        }
        vk_trace!("uploaded descriptor and input buffers");

        let desc_info = vk::DescriptorBufferInfo {
            buffer: desc_buffer.buffer,
            offset: 0,
            range: tensor_descs.len() as u64,
        };
        let input_info = vk::DescriptorBufferInfo {
            buffer: input_buffer.buffer,
            offset: 0,
            range: input_bytes.len() as u64,
        };
        let output_info = vk::DescriptorBufferInfo {
            buffer: output_buffer
                .as_ref()
                .map(|buf| buf.buffer)
                .unwrap_or(input_buffer.buffer),
            offset: 0,
            range: if output_alias_input {
                input_bytes.len() as u64
            } else {
                output_bytes.len() as u64
            },
        };
        let writes = [
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &desc_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &input_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 2,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &output_info,
                ..Default::default()
            },
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        let command_buffer = self.allocate_command_buffer()?;
        let begin_info = vk::CommandBufferBeginInfo {
            ..Default::default()
        };
        unsafe {
            self.device.begin_command_buffer(command_buffer, &begin_info)?;
            self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            if push_constants.len() >= 16 {
                let bytes = &push_constants[0..16];
                self.device.cmd_push_constants(
                    command_buffer,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes,
                );
            }
            let len = push_constants
                .get(0..4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap_or([0; 4])))
                .unwrap_or(0);
            let group_size = 256u32;
            let dispatch_x = (len + group_size - 1) / group_size;
            vk_trace!("dispatch len={} groups={}", len, dispatch_x.max(1));
            if let Some(query_pool) = self.query_pool {
                self.device
                    .cmd_reset_query_pool(command_buffer, query_pool, 0, 2);
                self.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    0,
                );
            }
            self.device.cmd_dispatch(command_buffer, dispatch_x.max(1), 1, 1);
            if let Some(query_pool) = self.query_pool {
                self.device.cmd_write_timestamp(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    query_pool,
                    1,
                );
            }
            self.device.end_command_buffer(command_buffer)?;
        }

        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            ..Default::default()
        };
        unsafe {
            self.device.queue_submit(self.queue, std::slice::from_ref(&submit_info), vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
            if let Some(query_pool) = self.query_pool {
                let mut data = [0u64; 2];
                self.device.get_query_pool_results(
                    query_pool,
                    0,
                    &mut data,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?;
                if data[1] > data[0] {
                    let delta = data[1] - data[0];
                    let nanos = (delta as f64 * self.timestamp_period as f64) as u64;
                    if let Ok(mut guard) = self.last_dispatch_ns.lock() {
                        *guard = Some(nanos);
                    }
                }
            }
            if let Some(output_buffer) = output_buffer {
                self.read_buffer(output_buffer, output_bytes)?;
            } else {
                self.read_buffer_range(input_buffer, output_offset, output_bytes)?;
            }
        }
        vk_trace!("dispatch complete and output readback finished");

        unsafe {
            self.device.destroy_pipeline(pipeline, None);
            self.device.destroy_shader_module(shader_module, None);
            self.destroy_buffer(desc_buffer);
            self.destroy_buffer(input_buffer);
            if let Some(output_buffer) = output_buffer {
                self.destroy_buffer(output_buffer);
            }
        }

        Ok(())
    }

    fn create_buffer(&self, size: u64) -> Result<BufferAlloc> {
        vk_trace!("create_buffer size={}", size);
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: vk::BufferUsageFlags::STORAGE_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };
        let mem_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_index = self.find_memory_type(
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_requirements.size,
            memory_type_index: mem_index,
            ..Default::default()
        };
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0)? };
        Ok(BufferAlloc { buffer, memory, size })
    }

    unsafe fn write_buffer(&self, buffer: BufferAlloc, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let ptr = self
            .device
            .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
        self.device.unmap_memory(buffer.memory);
        Ok(())
    }

    unsafe fn read_buffer(&self, buffer: BufferAlloc, output: &mut [u8]) -> Result<()> {
        if output.is_empty() {
            return Ok(());
        }
        let ptr = self
            .device
            .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())?;
        std::ptr::copy_nonoverlapping(ptr as *const u8, output.as_mut_ptr(), output.len());
        self.device.unmap_memory(buffer.memory);
        Ok(())
    }

    unsafe fn read_buffer_range(
        &self,
        buffer: BufferAlloc,
        offset: u64,
        output: &mut [u8],
    ) -> Result<()> {
        if output.is_empty() {
            return Ok(());
        }
        let ptr = self.device.map_memory(
            buffer.memory,
            offset,
            output.len() as u64,
            vk::MemoryMapFlags::empty(),
        )?;
        std::ptr::copy_nonoverlapping(ptr as *const u8, output.as_mut_ptr(), output.len());
        self.device.unmap_memory(buffer.memory);
        Ok(())
    }

    unsafe fn destroy_buffer(&self, buffer: BufferAlloc) {
        self.device.destroy_buffer(buffer.buffer, None);
        self.device.free_memory(buffer.memory, None);
    }

    fn allocate_command_buffer(&self) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: self.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        let buffers = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        Ok(buffers[0])
    }

    fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> Result<u32> {
        let mem_props =
            unsafe { self.instance.get_physical_device_memory_properties(self.physical_device) };
        for i in 0..mem_props.memory_type_count {
            let has_type = (type_filter & (1 << i)) != 0;
            let has_props = mem_props.memory_types[i as usize]
                .property_flags
                .contains(properties);
            if has_type && has_props {
                return Ok(i);
            }
        }
        Err(anyhow!("no suitable memory type found"))
    }
}

fn select_entrypoint<'a>(entrypoints: &'a [String], target: &'a str) -> &'a str {
    if entrypoints.iter().any(|name| name == target) {
        return target;
    }
    if entrypoints.iter().any(|name| name == "main") {
        return "main";
    }
    entrypoints.first().map(|s| s.as_str()).unwrap_or("main")
}

fn spirv_entrypoints(words: &[u32]) -> Vec<String> {
    const OP_ENTRY_POINT: u16 = 15;
    if words.len() < 5 {
        return Vec::new();
    }
    let mut offset = 5usize;
    let mut names = Vec::new();
    while offset < words.len() {
        let word = words[offset];
        let word_count = (word >> 16) as usize;
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 {
            break;
        }
        if opcode == OP_ENTRY_POINT {
            let operands = &words[(offset + 1)..(offset + word_count)];
            if operands.len() >= 3 {
                let (name, _) = read_spv_string(&operands[2..]);
                if !name.is_empty() {
                    names.push(name);
                }
            }
        }
        offset = offset.saturating_add(word_count);
    }
    names
}

fn read_spv_string(words: &[u32]) -> (String, usize) {
    let mut bytes = Vec::new();
    for (word_index, word) in words.iter().enumerate() {
        let raw = word.to_le_bytes();
        for &b in &raw {
            if b == 0 {
                let consumed = word_index + 1;
                let string = String::from_utf8_lossy(&bytes).to_string();
                return (string, consumed);
            }
            bytes.push(b);
        }
    }
    (String::from_utf8_lossy(&bytes).to_string(), words.len())
}

fn validate_spirv(bytes: &[u8], target: &str) -> Result<()> {
    if bytes.len() < 20 || bytes.len() % 4 != 0 {
        return Err(anyhow!(
            "invalid SPIR-V size for {} ({} bytes)",
            target,
            bytes.len()
        ));
    }
    let words = bytemuck::cast_slice::<u8, u32>(bytes);
    let magic = words[0];
    let version = words[1];
    let bound = words[3];
    let schema = words[4];
    vk_trace!(
        "spv header target={} magic=0x{:08x} version=0x{:08x} bound={} schema={}",
        target,
        magic,
        version,
        bound,
        schema
    );
    if magic != 0x0723_0203 {
        return Err(anyhow!(
            "invalid SPIR-V magic for {} (0x{:08x})",
            target,
            magic
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct BufferAlloc {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

static VULKAN_RUNTIME: OnceLock<VulkanRuntime> = OnceLock::new();

#[allow(dead_code)]
pub fn set_vulkan_runtime(runtime: VulkanRuntime) -> Result<()> {
    VULKAN_RUNTIME
        .set(runtime)
        .map_err(|_| anyhow!("vulkan runtime already initialized"))
}

pub fn get_vulkan_runtime() -> Option<&'static VulkanRuntime> {
    VULKAN_RUNTIME.get()
}

pub fn take_last_dispatch_duration() -> Option<Duration> {
    get_vulkan_runtime()?.take_last_dispatch_duration()
}
