use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes;
use crate::backend::VulkanBuffer;
use crate::graph::OpKind;

pub mod registry;

pub fn add_generic(a: &VulkanBuffer, b: &VulkanBuffer) -> Result<VulkanBuffer> {
    let strict_shapes = a.shader_setting_bool("strict_shapes").unwrap_or(true);
    if strict_shapes && a.len != b.len {
        return Err(anyhow!("add op shape mismatch"));
    }
    let allow_mixed = a.shader_setting_bool("allow_mixed_dtypes").unwrap_or(false);
    if !allow_mixed && a.dtype != b.dtype {
        return Err(anyhow!("add op expects matching dtypes"));
    }
    let len = if strict_shapes {
        a.len
    } else {
        a.len.min(b.len)
    };
    let runtime = super::runtime_from_buffers(a, Some(b))?;
    let entry = super::entry_point_name("add", a.dtype)?;
    let spirv = a
        .spv_bytes_for_dtype(a.dtype)
        .ok_or_else(|| anyhow!("missing SPIR-V for add op"))?;
    let output_size = storage_size_bytes(a.dtype) * len;
    let output_inner = runtime.create_buffer(output_size)?;
    runtime.dispatch(
        OpKind::Add,
        a.dtype,
        entry,
        spirv,
        &a.inner,
        &b.inner,
        &output_inner,
        0,
        len,
    )?;
    Ok(VulkanBuffer {
        dtype: a.dtype,
        len,
        shader: a.shader.clone(),
        inner: output_inner,
    })
}
