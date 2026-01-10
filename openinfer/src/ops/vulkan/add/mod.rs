use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes;
use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::graph::OpKind;
use crate::tensor::{compute_strides, DType};
use crate::timer::Timer;

pub mod registry;

pub fn add_generic(attrs: &OpAttrs, a: &VulkanBuffer, b: &VulkanBuffer, thread_id: usize) -> Result<VulkanBuffer> {
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
    let target = super::spv_target_name(OpKind::Add, a.dtype, attrs)?;
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for add op", target))?;
    let output_size = storage_size_bytes(a.dtype) * len;
    let output_inner = runtime.create_buffer(output_size)?;
    let push = [len as u32, 0, 0, 0];
    let duration_ns = runtime.dispatch(
        OpKind::Add,
        a.dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &b.inner,
        &output_inner,
        push,
        len,
    )?;
    Timer::record(thread_id, duration_ns);
    let shape = if len == a.len { a.shape.clone() } else { vec![len] };
    let strides = compute_strides(shape.as_slice());
    Ok(VulkanBuffer {
        dtype: a.dtype,
        len,
        shape,
        strides,
        shader: a.shader.clone(),
        inner: output_inner,
    })
}

pub(crate) fn spv_target_name_add(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::I8, &OpAttrs::None)
        | (DType::I16, &OpAttrs::None)
        | (DType::F32, &OpAttrs::None)
        | (DType::Bool, &OpAttrs::None)
        | (DType::U8, &OpAttrs::None)
        | (DType::U16, &OpAttrs::None)
        | (DType::I32, &OpAttrs::None)
        | (DType::U32, &OpAttrs::None)
        | (DType::I64, &OpAttrs::None)
        | (DType::U64, &OpAttrs::None) => Ok(format!("add_{}", super::dtype_suffix(dtype).unwrap())),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for add dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}
