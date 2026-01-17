use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use ash::{vk, Entry};

use crate::graph::OpKind;
use crate::tensor::DType;

#[allow(dead_code)]
pub struct VulkanRuntime {
    entry: Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipelines: Mutex<HashMap<(OpKind, DType, String), vk::Pipeline>>,
    supports_i64: bool,
    supports_timestamps: bool,
    timestamp_period: f32,
    max_descriptor_sets: u32,
}

// VulkanRuntime is shared across threads behind external synchronization.
unsafe impl Send for VulkanRuntime {}
unsafe impl Sync for VulkanRuntime {}

const MIN_DESCRIPTOR_SETS: u32 = 1;

pub struct VulkanBufferInner {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    runtime: Arc<VulkanRuntime>,
}

impl VulkanRuntime {
    pub fn new() -> Result<Self> {
        let entry = unsafe { Entry::load()? };
        let trace = std::env::var("OPENINFER_VULKAN_TRACE").is_ok();
        if trace {
            eprintln!("vulkan init: creating instance");
        }
        let app_name = CString::new("openinfer").unwrap_or_else(|_| CString::new("openinfer").expect("static"));
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&app_name)
            .engine_version(0)
            .api_version(vk::API_VERSION_1_1);

        let instance_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        if trace {
            eprintln!("vulkan init: enumerate physical devices");
        }
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let (physical_device, queue_family_index) = physical_devices
            .iter()
            .find_map(|device| pick_compute_queue(&instance, *device))
            .ok_or_else(|| anyhow!("no Vulkan compute queue found"))?;

        if trace {
            eprintln!("vulkan init: create device");
        }
        let priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);
        let supported = unsafe { instance.get_physical_device_features(physical_device) };
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let mut enabled = vk::PhysicalDeviceFeatures::default();
        enabled.shader_int64 = supported.shader_int64;
        enabled.shader_float64 = supported.shader_float64;
        let max_descriptor_sets = props.limits.max_bound_descriptor_sets.max(MIN_DESCRIPTOR_SETS);
        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_features(&enabled);
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };
        if trace {
            eprintln!("vulkan init: create command pool");
        }

        let descriptor_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        }];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(1)
            .pool_sizes(&descriptor_pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_info, None)? };
        if trace {
            eprintln!("vulkan init: create descriptor pool");
        }

        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
            vk::DescriptorSetLayoutBinding::builder()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build(),
        ];
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&set_layout_info, None)? };
        if trace {
            eprintln!("vulkan init: create descriptor set layout");
        }

        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)
            .build();
        let set_layouts = vec![descriptor_set_layout; max_descriptor_sets as usize];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(std::slice::from_ref(&push_constant_range));
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };
        if trace {
            eprintln!(
                "vulkan init: create pipeline layout (set_layouts={})",
                set_layouts.len()
            );
        }

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            command_pool,
            descriptor_pool,
            descriptor_set_layout,
            pipeline_layout,
            pipelines: Mutex::new(HashMap::new()),
            supports_i64: supported.shader_int64 != 0,
            supports_timestamps: props.limits.timestamp_compute_and_graphics != 0,
            timestamp_period: props.limits.timestamp_period,
            max_descriptor_sets,
        })
    }

    pub fn supports_i64(&self) -> bool {
        self.supports_i64
    }


    pub fn create_buffer(self: &Arc<Self>, size: usize) -> Result<Arc<VulkanBufferInner>> {
        let buffer_size = size.max(4);
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(buffer_size as vk::DeviceSize)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type_index = find_memory_type(
            &self.instance,
            self.physical_device,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let allocate_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { self.device.allocate_memory(&allocate_info, None)? };
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0)? };

        Ok(Arc::new(VulkanBufferInner {
            buffer,
            memory,
            size: buffer_size as vk::DeviceSize,
            runtime: Arc::clone(self),
        }))
    }

    pub fn write_buffer(&self, buffer: &VulkanBufferInner, data: &[u8]) -> Result<()> {
        if data.len() as vk::DeviceSize > buffer.size {
            return Err(anyhow!("upload exceeds Vulkan buffer size"));
        }
        if data.is_empty() {
            return Ok(());
        }
        unsafe {
            let ptr = self
                .device
                .map_memory(buffer.memory, 0, data.len() as vk::DeviceSize, vk::MemoryMapFlags::empty())?;
            if ptr.is_null() {
                return Err(anyhow!("vulkan map_memory returned null pointer"));
            }
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<u8>(), data.len());
            let range = vk::MappedMemoryRange::builder()
                .memory(buffer.memory)
                .offset(0)
                .size(data.len() as vk::DeviceSize);
            self.device.flush_mapped_memory_ranges(std::slice::from_ref(&range))?;
            self.device.unmap_memory(buffer.memory);
        }
        Ok(())
    }

    pub fn read_buffer(&self, buffer: &VulkanBufferInner, data: &mut [u8]) -> Result<()> {
        if data.len() as vk::DeviceSize > buffer.size {
            return Err(anyhow!("download exceeds Vulkan buffer size"));
        }
        if data.is_empty() {
            return Ok(());
        }
        unsafe {
            let ptr = self
                .device
                .map_memory(buffer.memory, 0, data.len() as vk::DeviceSize, vk::MemoryMapFlags::empty())?;
            if ptr.is_null() {
                return Err(anyhow!("vulkan map_memory returned null pointer"));
            }
            let range = vk::MappedMemoryRange::builder()
                .memory(buffer.memory)
                .offset(0)
                .size(data.len() as vk::DeviceSize);
            self.device.invalidate_mapped_memory_ranges(std::slice::from_ref(&range))?;
            std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), data.as_mut_ptr(), data.len());
            self.device.unmap_memory(buffer.memory);
        }
        Ok(())
    }

    #[allow(unused)]
    pub fn dispatch(
        &self,
        op: OpKind,
        dtype: DType,
        pipeline_key: &str,
        entry_point: &str,
        spirv: &[u8],
        input0: &VulkanBufferInner,
        input1: &VulkanBufferInner,
        output: &VulkanBufferInner,
        push: [u32; 4],
        len: usize,
    ) -> Result<u128> {
        if len == 0 {
            return Ok(0);
        }
        let set_index = descriptor_set_index_from_spirv(spirv).unwrap_or(0);
        if set_index >= self.max_descriptor_sets {
            return Err(anyhow!(
                "vulkan descriptor set index {} exceeds max {}",
                set_index,
                self.max_descriptor_sets
            ));
        }
        if std::env::var("OPENINFER_VULKAN_TRACE").is_ok() {
            eprintln!(
                "vulkan dispatch op={} dtype={:?} entry={} len={} push={:?} set={}",
                op.as_str(),
                dtype,
                entry_point,
                len,
                push,
                set_index
            );
        }
        let pipeline = self.pipeline_for_op(op, dtype, pipeline_key, entry_point, spirv)?;
        let descriptor_set = self.allocate_descriptor_set()?;

        struct QueryPoolGuard<'a> {
            device: &'a ash::Device,
            pool: vk::QueryPool,
        }

        impl<'a> Drop for QueryPoolGuard<'a> {
            fn drop(&mut self) {
                unsafe {
                    self.device.destroy_query_pool(self.pool, None);
                }
            }
        }

        let query_pool = if self.supports_timestamps {
            let info = vk::QueryPoolCreateInfo::builder()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2);
            Some(unsafe { self.device.create_query_pool(&info, None)? })
        } else {
            None
        };
        let _query_pool_guard = query_pool.map(|pool| QueryPoolGuard {
            device: &self.device,
            pool,
        });

        let input1_buffer = if std::ptr::eq(input0, input1) {
            output
        } else {
            input1
        };
        let buffer_infos = [
            vk::DescriptorBufferInfo {
                buffer: input0.buffer,
                offset: 0,
                range: input0.size,
            },
            vk::DescriptorBufferInfo {
                buffer: input1_buffer.buffer,
                offset: 0,
                range: input1_buffer.size,
            },
            vk::DescriptorBufferInfo {
                buffer: output.buffer,
                offset: 0,
                range: output.size,
            },
        ];
        let writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[0]))
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[1]))
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[2]))
                .build(),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        let command_buffer = self.allocate_command_buffer()?;
        let begin_info = vk::CommandBufferBeginInfo::builder();
        unsafe { self.device.begin_command_buffer(command_buffer, &begin_info)? };

        unsafe {
            let pre_barrier = vk::MemoryBarrier::builder()
                .src_access_mask(
                    vk::AccessFlags::HOST_WRITE | vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                )
                .build();
            if let Some(pool) = query_pool {
                self.device.cmd_reset_query_pool(command_buffer, pool, 0, 2);
            }
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&pre_barrier),
                &[],
                &[],
            );
            self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                set_index,
                std::slice::from_ref(&descriptor_set),
                &[],
            );
            let push_bytes = std::slice::from_raw_parts(push.as_ptr().cast::<u8>(), 16);
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );
            if let Some(pool) = query_pool {
                self.device
                    .cmd_write_timestamp(command_buffer, vk::PipelineStageFlags::COMPUTE_SHADER, pool, 0);
            }
            let groups = (len as u32 + 255) / 256;
            self.device.cmd_dispatch(command_buffer, groups, 1, 1);
            if let Some(pool) = query_pool {
                self.device
                    .cmd_write_timestamp(command_buffer, vk::PipelineStageFlags::COMPUTE_SHADER, pool, 1);
            }
            let post_barrier = vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .build();
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                std::slice::from_ref(&post_barrier),
                &[],
                &[],
            );
            self.device.end_command_buffer(command_buffer)?;
        }

        let submit_info = vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));
        unsafe {
            self.device.queue_submit(self.queue, std::slice::from_ref(&submit_info), vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
        }

        let duration_ns = if let Some(pool) = query_pool {
            let mut timestamps = [0u64; 2];
            unsafe {
                self.device.get_query_pool_results(
                    pool,
                    0,
                    2,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )?;
            }
            let delta = timestamps[1].saturating_sub(timestamps[0]) as f64;
            (delta * self.timestamp_period as f64) as u128
        } else {
            0
        };

        unsafe {
            self.device.free_command_buffers(self.command_pool, std::slice::from_ref(&command_buffer));
            self.device.free_descriptor_sets(self.descriptor_pool, std::slice::from_ref(&descriptor_set))?;
        }

        Ok(duration_ns)
    }

    fn allocate_descriptor_set(&self) -> Result<vk::DescriptorSet> {
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
        let descriptor_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? };
        descriptor_sets
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("failed to allocate descriptor set"))
    }

    fn allocate_command_buffer(&self) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let buffers = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        buffers
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("failed to allocate command buffer"))
    }

    fn pipeline_for_op(
        &self,
        op: OpKind,
        dtype: DType,
        pipeline_key: &str,
        entry_point: &str,
        spirv: &[u8],
    ) -> Result<vk::Pipeline> {
        let mut pipelines = self.pipelines.lock().map_err(|_| anyhow!("pipeline mutex poisoned"))?;
        let key = (op, dtype, pipeline_key.to_string());
        if let Some(pipeline) = pipelines.get(&key) {
            return Ok(*pipeline);
        }

        let shader_module = create_shader_module(&self.device, spirv)?;
        let entry = CString::new(entry_point)
            .map_err(|_| anyhow!("invalid Vulkan entry point {}", entry_point))?;
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry)
            .build();
        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_info)
            .layout(self.pipeline_layout);
        let pipeline = unsafe {
            let result = self
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None)
                .map_err(|e| anyhow!("failed to create compute pipeline: {:?}", e))?;
            result[0]
        };
        unsafe { self.device.destroy_shader_module(shader_module, None) };
        pipelines.insert(key, pipeline);
        Ok(pipeline)
    }
}

#[allow(unused)]
impl VulkanBufferInner {
    pub fn runtime(&self) -> &Arc<VulkanRuntime> {
        &self.runtime
    }
}

impl std::fmt::Debug for VulkanRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanRuntime").finish()
    }
}

impl std::fmt::Debug for VulkanBufferInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBufferInner")
            .field("size", &self.size)
            .finish()
    }
}

impl Drop for VulkanBufferInner {
    fn drop(&mut self) {
        unsafe {
            self.runtime.device.destroy_buffer(self.buffer, None);
            self.runtime.device.free_memory(self.memory, None);
        }
    }
}

impl Drop for VulkanRuntime {
    fn drop(&mut self) {
        unsafe {
            for pipeline in self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).values() {
                self.device.destroy_pipeline(*pipeline, None);
            }
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

pub fn storage_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::I64 | DType::U64 | DType::F64 => 8,
        _ => 4,
    }
}

fn pick_compute_queue(instance: &ash::Instance, device: vk::PhysicalDevice) -> Option<(vk::PhysicalDevice, u32)> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    families
        .iter()
        .enumerate()
        .find(|(_, family)| family.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .map(|(index, _)| (device, index as u32))
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Result<u32> {
    let props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for i in 0..props.memory_type_count {
        if (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize].property_flags.contains(flags)
        {
            return Ok(i);
        }
    }
    Err(anyhow!("failed to find suitable Vulkan memory type"))
}

fn create_shader_module(device: &ash::Device, spirv: &[u8]) -> Result<vk::ShaderModule> {
    let mut cursor = std::io::Cursor::new(spirv);
    let code = ash::util::read_spv(&mut cursor)?;
    let info = vk::ShaderModuleCreateInfo::builder().code(&code);
    let module = unsafe { device.create_shader_module(&info, None)? };
    Ok(module)
}

fn descriptor_set_index_from_spirv(spirv: &[u8]) -> Option<u32> {
    if spirv.len() < 20 || spirv.len() % 4 != 0 {
        return None;
    }
    let mut words = Vec::with_capacity(spirv.len() / 4);
    for chunk in spirv.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let mut idx = 5;
    while idx < words.len() {
        let word = words[idx];
        let op = word & 0xFFFF;
        let count = (word >> 16) as usize;
        if count == 0 || idx + count > words.len() {
            break;
        }
        if op == 71 {
            if idx + 3 < words.len() {
                let decoration = words[idx + 2];
                if decoration == 34 {
                    return Some(words[idx + 3]);
                }
            }
        }
        idx += count;
    }
    None
}
