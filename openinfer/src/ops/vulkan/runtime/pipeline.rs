use anyhow::{anyhow, Result};
use ash::vk;

use crate::vk_trace;

use super::VulkanRuntime;

impl VulkanRuntime {
    pub(super) fn create_descriptor_set_layout(
        &self,
        binding_count: usize,
    ) -> Result<vk::DescriptorSetLayout> {
        let mut bindings = Vec::with_capacity(binding_count);
        for binding in 0..binding_count {
            bindings.push(vk::DescriptorSetLayoutBinding {
                binding: binding as u32,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            });
        }
        let set_layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let layout = unsafe { self.device.create_descriptor_set_layout(&set_layout_info, None)? };
        Ok(layout)
    }

    pub(super) fn create_pipeline_layout(
        &self,
        set_layout: vk::DescriptorSetLayout,
        push_constant_size: u32,
    ) -> Result<vk::PipelineLayout> {
        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: push_constant_size,
        };
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &set_layout,
            push_constant_range_count: if push_constant_size > 0 { 1 } else { 0 },
            p_push_constant_ranges: if push_constant_size > 0 {
                &push_range
            } else {
                std::ptr::null()
            },
            ..Default::default()
        };
        let pipeline_layout =
            unsafe { self.device.create_pipeline_layout(&pipeline_layout_info, None)? };
        Ok(pipeline_layout)
    }

    pub(super) fn create_descriptor_pool(&self, binding_count: usize) -> Result<vk::DescriptorPool> {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: binding_count as u32,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        let pool = unsafe { self.device.create_descriptor_pool(&pool_info, None)? };
        Ok(pool)
    }

    pub(super) fn create_shader_module(&self, spv_bytes: &[u8]) -> Result<vk::ShaderModule> {
        let spv_u32 = bytemuck::cast_slice::<u8, u32>(spv_bytes);
        let shader_info = vk::ShaderModuleCreateInfo {
            code_size: spv_bytes.len(),
            p_code: spv_u32.as_ptr(),
            ..Default::default()
        };
        let module = unsafe { self.device.create_shader_module(&shader_info, None)? };
        Ok(module)
    }

    pub(super) fn create_compute_pipeline(
        &self,
        module: vk::ShaderModule,
        entry_name: &std::ffi::CString,
        pipeline_layout: vk::PipelineLayout,
    ) -> Result<vk::Pipeline> {
        let stage_info = vk::PipelineShaderStageCreateInfo {
            stage: vk::ShaderStageFlags::COMPUTE,
            module,
            p_name: entry_name.as_ptr(),
            ..Default::default()
        };
        let pipeline_info = vk::ComputePipelineCreateInfo {
            stage: stage_info,
            layout: pipeline_layout,
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
        Ok(pipeline)
    }
}
