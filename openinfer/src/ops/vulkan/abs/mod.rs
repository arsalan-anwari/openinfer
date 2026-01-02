use anyhow::Result;

use crate::backend::vulkan::storage_size_bytes;
use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::graph::OpKind;
use crate::tensor::DType;
use anyhow::anyhow;

pub mod registry;

pub fn abs_generic(attrs: &OpAttrs, a: &VulkanBuffer) -> Result<VulkanBuffer> {
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
    let target = super::spv_target_name(OpKind::Abs, a.dtype, attrs)?;
    let entry = super::entry_point_name();
    let output_size = storage_size_bytes(a.dtype) * a.len;
    let output_inner = runtime.create_buffer(output_size)?;
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow::anyhow!("missing SPIR-V target {} for abs op", target))?;
    runtime.dispatch(
        OpKind::Abs,
        a.dtype,
        &target,
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

pub(crate) fn spv_target_name_abs(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::I8, OpAttrs::None)
        | (DType::I16, OpAttrs::None)
        | (DType::F32, OpAttrs::None)
        | (DType::I32, OpAttrs::None)
        | (DType::I64, OpAttrs::None) => Ok(format!("abs_{}", super::dtype_suffix(dtype).unwrap())),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for abs dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}
