use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes_for_len;
use crate::backend::VulkanBuffer;
use crate::graph::{OpAttrs, OpKind};
use crate::tensor::{compute_strides, DType};
use crate::timer::Timer;

pub mod registry;

fn matmul_dims(a: &VulkanBuffer, b: &VulkanBuffer) -> Result<(usize, usize, usize)> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(anyhow!(
            "matmul expects 2D inputs, got {:?} and {:?}",
            a.shape,
            b.shape
        ));
    }
    let m = a.shape[0];
    let k = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];
    if k != k2 {
        return Err(anyhow!(
            "matmul inner dims must match, got {:?} and {:?}",
            a.shape,
            b.shape
        ));
    }
    Ok((m, k, n))
}

pub fn matmul_generic(
    attrs: &OpAttrs,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    thread_id: usize,
) -> Result<VulkanBuffer> {
    if a.effective_dtype != b.effective_dtype {
        return Err(anyhow!("matmul op expects matching dtypes"));
    }
    let (m, k, n) = matmul_dims(a, b)?;
    let runtime = super::runtime_from_buffers(a, Some(b))?;
    let target = super::spv_target_name(OpKind::Matmul, a.effective_dtype, attrs)?;
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for matmul op", target))?;
    let len = m.saturating_mul(n);
    let output_size = storage_size_bytes_for_len(a.effective_dtype, len);
    let output_inner = runtime.create_buffer(output_size)?;
    let push = [len as u32, m as u32, n as u32, k as u32];
    let duration_ns = runtime.dispatch(
        OpKind::Matmul,
        a.effective_dtype,
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
    let shape = vec![m, n];
    let strides = compute_strides(shape.as_slice());
    Ok(VulkanBuffer {
        dtype: a.dtype,
        effective_dtype: a.effective_dtype,
        len,
        shape,
        strides,
        shader: a.shader.clone(),
        inner: output_inner,
    })
}

pub(crate) fn spv_target_name_matmul(dtype: DType, attrs: &OpAttrs) -> Result<String> {
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
        | (DType::U64, &OpAttrs::None) => {
            Ok(format!("matmul_{}", super::dtype_suffix(dtype).unwrap()))
        }
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for matmul dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}
