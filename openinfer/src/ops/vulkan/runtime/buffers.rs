use anyhow::{anyhow, Result};
use ash::vk;

use crate::vk_trace;

use super::VulkanRuntime;

#[derive(Clone, Copy)]
pub(super) struct BufferAlloc {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: u64,
}

impl VulkanRuntime {
    pub(super) fn allocate_command_buffer(&self) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: self.command_pool,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: 1,
            ..Default::default()
        };
        let buffers = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        Ok(buffers[0])
    }

    pub(super) fn create_buffer(&self, size: u64) -> Result<BufferAlloc> {
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

    pub(super) unsafe fn write_buffer(&self, buffer: BufferAlloc, bytes: &[u8]) -> Result<()> {
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

    pub(super) unsafe fn read_buffer(&self, buffer: BufferAlloc, output: &mut [u8]) -> Result<()> {
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

    pub(super) unsafe fn read_buffer_range(
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

    pub(super) unsafe fn destroy_buffer(&self, buffer: BufferAlloc) {
        self.device.destroy_buffer(buffer.buffer, None);
        self.device.free_memory(buffer.memory, None);
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
