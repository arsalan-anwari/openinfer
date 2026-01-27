use std::ffi::CString;
use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use ash::vk;

use crate::ops::vulkan::dispatch::{BindingBytes, VulkanOpSpec};
use crate::ops::vulkan::spv::embedded_spv;
use crate::vk_trace;

use super::buffers::BufferAlloc;
use super::VulkanRuntime;

impl VulkanRuntime {
    pub fn dispatch_compute(
    &self,
    spec: &VulkanOpSpec<'_>,
    bindings: &mut [BindingBytes<'_>],
    push_constants: &[u8],
    dispatch_len: u32,
) -> Result<()> {
    if push_constants.len() > spec.push_constant_size as usize {
        return Err(anyhow!(
            "push constants too large ({} > {})",
            push_constants.len(),
            spec.push_constant_size
        ));
    }
    if let Ok(mut guard) = self.last_dispatch_ns.lock() {
        *guard = None;
    }
    let spv_path = PathBuf::from(spec.spv_dir).join(format!("{}.spv", spec.entry));
    vk_trace!(
        "dispatch_compute entry={} bindings={} push_bytes={}",
        spec.entry,
        bindings.len(),
        push_constants.len()
    );
    let spv_bytes = if let Some(bytes) = embedded_spv(spec.entry) {
        bytes.to_vec()
    } else {
        let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(&spv_path);
        fs::read(&spv_path)
            .or_else(|_| fs::read(&manifest_path))
            .map_err(|err| anyhow!("failed to read {}: {}", spv_path.display(), err))?
    };
    self.validate_spirv(&spv_bytes, spec.entry)?;
    let spv_u32 = bytemuck::cast_slice::<u8, u32>(&spv_bytes);
    let entrypoints = self.spirv_entrypoints(spv_u32);
    if !entrypoints.is_empty() {
        vk_trace!("spv entrypoints: {:?}", entrypoints);
    }
    let shader_module = self.create_shader_module(&spv_bytes)?;
    let entry_name = self.select_entrypoint(&entrypoints, spec.entry);
    let entry_name = CString::new(entry_name)?;
    let descriptor_set_layout = self.create_descriptor_set_layout(bindings.len())?;
    let pipeline_layout =
        self.create_pipeline_layout(descriptor_set_layout, spec.push_constant_size)?;
    let pipeline = self.create_compute_pipeline(shader_module, &entry_name, pipeline_layout)?;

    let descriptor_pool = self.create_descriptor_pool(bindings.len())?;
    let set_alloc_info = vk::DescriptorSetAllocateInfo {
        descriptor_pool,
        descriptor_set_count: 1,
        p_set_layouts: &descriptor_set_layout,
        ..Default::default()
    };
    let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&set_alloc_info)?[0] };
    vk_trace!("allocated descriptor set");

    let mut buffers: Vec<Option<BufferAlloc>> = vec![None; bindings.len()];
    for (index, binding) in bindings.iter_mut().enumerate() {
        match binding {
            BindingBytes::ReadOnly(bytes) => {
                let buffer = self.create_buffer(bytes.len() as u64)?;
                unsafe { self.write_buffer(buffer, bytes)? };
                buffers[index] = Some(buffer);
            }
            BindingBytes::ReadWrite(bytes) => {
                let buffer = self.create_buffer(bytes.len() as u64)?;
                unsafe { self.write_buffer(buffer, bytes)? };
                buffers[index] = Some(buffer);
            }
            BindingBytes::Alias { .. } => {}
        }
    }
    vk_trace!("created buffers for bindings");

    let mut writes = Vec::with_capacity(bindings.len());
    let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(bindings.len());
    for (index, binding) in bindings.iter_mut().enumerate() {
        let (buffer, range) = match binding {
            BindingBytes::ReadOnly(bytes) => {
                let buffer = buffers[index].ok_or_else(|| anyhow!("missing buffer {}", index))?;
                (buffer, bytes.len() as u64)
            }
            BindingBytes::ReadWrite(bytes) => {
                let buffer = buffers[index].ok_or_else(|| anyhow!("missing buffer {}", index))?;
                (buffer, bytes.len() as u64)
            }
            BindingBytes::Alias {
                source_binding, ..
            } => {
                let buffer = buffers
                    .get(*source_binding)
                    .and_then(|buf| *buf)
                    .ok_or_else(|| anyhow!("missing alias source buffer {}", source_binding))?;
                (buffer, buffer.size)
            }
        };
        buffer_infos.push(vk::DescriptorBufferInfo {
            buffer: buffer.buffer,
            offset: 0,
            range,
        });
        writes.push(vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: index as u32,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            p_buffer_info: buffer_infos.last().unwrap(),
            ..Default::default()
        });
    }
    unsafe { self.device.update_descriptor_sets(&writes, &[]) };

    let command_buffer = self.allocate_command_buffer()?;
    let begin_info = vk::CommandBufferBeginInfo {
        ..Default::default()
    };
    unsafe {
        self.device.begin_command_buffer(command_buffer, &begin_info)?;
        self.device
            .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            std::slice::from_ref(&descriptor_set),
            &[],
        );
        if spec.push_constant_size > 0 && !push_constants.is_empty() {
            self.device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants,
            );
        }
        let group_size = spec.workgroup_size[0].max(1);
        let dispatch_x = (dispatch_len + group_size - 1) / group_size;
        vk_trace!("dispatch len={} groups={}", dispatch_len, dispatch_x.max(1));
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
        self.device
            .cmd_dispatch(command_buffer, dispatch_x.max(1), 1, 1);
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
        self.device.queue_submit(
            self.queue,
            std::slice::from_ref(&submit_info),
            vk::Fence::null(),
        )?;
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
        for (index, binding) in bindings.iter_mut().enumerate() {
            match binding {
                BindingBytes::ReadOnly(_) => {}
                BindingBytes::ReadWrite(bytes) => {
                    let buffer = buffers[index].ok_or_else(|| anyhow!("missing buffer {}", index))?;
                    self.read_buffer(buffer, bytes)?;
                }
                BindingBytes::Alias {
                    source_binding,
                    offset,
                    bytes,
                } => {
                    let buffer = buffers
                        .get(*source_binding)
                        .and_then(|buf| *buf)
                        .ok_or_else(|| anyhow!("missing alias source buffer {}", source_binding))?;
                    self.read_buffer_range(buffer, *offset, bytes)?;
                }
            }
        }
    }
    vk_trace!("dispatch complete and output readback finished");

    unsafe {
        self.device.destroy_pipeline(pipeline, None);
        self.device.destroy_shader_module(shader_module, None);
        self.device
            .destroy_descriptor_set_layout(descriptor_set_layout, None);
        self.device.destroy_pipeline_layout(pipeline_layout, None);
        self.device.destroy_descriptor_pool(descriptor_pool, None);
        for buffer in buffers.into_iter().flatten() {
            self.destroy_buffer(buffer);
        }
    }

    Ok(())
}
}
