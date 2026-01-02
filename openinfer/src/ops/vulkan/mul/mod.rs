use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes;
use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::graph::OpKind;
use crate::tensor::DType;

pub mod registry;

pub fn mul_generic(attrs: &OpAttrs, a: &VulkanBuffer, b: &VulkanBuffer) -> Result<VulkanBuffer> {
    let strict_shapes = a.shader_setting_bool("strict_shapes").unwrap_or(true);
    if strict_shapes && a.len != b.len {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let allow_mixed = a.shader_setting_bool("allow_mixed_dtypes").unwrap_or(false);
    if !allow_mixed && a.dtype != b.dtype {
        return Err(anyhow!("mul op expects matching dtypes"));
    }
    let len = if strict_shapes {
        a.len
    } else {
        a.len.min(b.len)
    };
    let runtime = super::runtime_from_buffers(a, Some(b))?;
    let target = super::spv_target_name(OpKind::Mul, a.dtype, attrs)?;
    let entry = super::entry_point_name();
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for mul op", target))?;
    let output_size = storage_size_bytes(a.dtype) * len;
    let output_inner = runtime.create_buffer(output_size)?;
    runtime.dispatch(
        OpKind::Mul,
        a.dtype,
        &target,
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

pub(crate) fn spv_target_name_mul(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::I8, OpAttrs::None)
        | (DType::I16, OpAttrs::None)
        | (DType::F32, OpAttrs::None)
        | (DType::Bool, OpAttrs::None)
        | (DType::U8, OpAttrs::None)
        | (DType::U16, OpAttrs::None)
        | (DType::I32, OpAttrs::None)
        | (DType::U32, OpAttrs::None)
        | (DType::I64, OpAttrs::None)
        | (DType::U64, OpAttrs::None) => Ok(format!("mul_{}", super::dtype_suffix(dtype).unwrap())),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for mul dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}
