use anyhow::Result;

use crate::backend::vulkan::storage_size_bytes;
use crate::backend::VulkanBuffer;
use crate::graph::OpKind;

pub mod registry;

pub fn abs_generic(a: &VulkanBuffer) -> Result<VulkanBuffer> {
    let runtime = super::runtime_from_buffers(a, None)?;
    let unsigned_identity = a.shader_setting_bool("abs_unsigned_is_identity").unwrap_or(true);
    if unsigned_identity
        && matches!(
            a.dtype,
            crate::tensor::DType::U8
                | crate::tensor::DType::U16
                | crate::tensor::DType::U32
                | crate::tensor::DType::U64
                | crate::tensor::DType::Bool
        )
    {
        return Ok(VulkanBuffer {
            dtype: a.dtype,
            len: a.len,
            shader: a.shader.clone(),
            inner: a.inner.clone(),
        });
    }
    let flags = if unsigned_identity { 1 } else { 0 };
    let entry = super::entry_point_name("abs", a.dtype)?;
    let output_size = storage_size_bytes(a.dtype) * a.len;
    let output_inner = runtime.create_buffer(output_size)?;
    let spirv = a
        .spv_bytes_for_dtype(a.dtype)
        .ok_or_else(|| anyhow::anyhow!("missing SPIR-V for abs op"))?;
    runtime.dispatch(
        OpKind::Abs,
        a.dtype,
        entry,
        spirv,
        &a.inner,
        &a.inner,
        &output_inner,
        flags,
        a.len,
    )?;
    Ok(VulkanBuffer {
        dtype: a.dtype,
        len: a.len,
        shader: a.shader.clone(),
        inner: output_inner,
    })
}
