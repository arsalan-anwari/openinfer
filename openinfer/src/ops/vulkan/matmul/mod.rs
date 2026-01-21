use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes_for_len;
use crate::backend::VulkanBuffer;
use crate::graph::{OpAttrs, OpKind};
use crate::tensor::{compute_strides, DType};
use crate::timer::Timer;

pub mod registry;
pub mod registry_inplace;

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

pub fn matmul_accumulate_generic(
    attrs: &OpAttrs,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    output_dtype: DType,
    thread_id: usize,
) -> Result<VulkanBuffer> {
    if a.effective_dtype != b.effective_dtype {
        return Err(anyhow!("matmul accumulate expects matching dtypes"));
    }
    let (m, k, n) = matmul_dims(a, b)?;
    let runtime = super::runtime_from_buffers(a, Some(b))?;
    let target = spv_target_name_matmul_accumulate(a.effective_dtype, output_dtype, attrs)?;
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for matmul accumulate", target))?;
    let len = m.saturating_mul(n);
    let output_size = storage_size_bytes_for_len(output_dtype, len);
    let output_inner = runtime.create_buffer(output_size)?;
    let push = [len as u32, m as u32, n as u32, k as u32];
    let duration_ns = runtime.dispatch(
        OpKind::Matmul,
        output_dtype,
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
        dtype: output_dtype,
        effective_dtype: output_dtype,
        len,
        shape,
        strides,
        shader: a.shader.clone(),
        inner: output_inner,
    })
}

pub fn matmul_inplace_generic(
    attrs: &OpAttrs,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    thread_id: usize,
) -> Result<VulkanBuffer> {
    matmul_generic(attrs, a, b, thread_id)
}

pub(crate) fn spv_target_name_matmul(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::I8, &OpAttrs::None)
        | (DType::I16, &OpAttrs::None)
        | (DType::I32, &OpAttrs::None)
        | (DType::I64, &OpAttrs::None)
        | (DType::U8, &OpAttrs::None)
        | (DType::U16, &OpAttrs::None)
        | (DType::U32, &OpAttrs::None)
        | (DType::U64, &OpAttrs::None)
        | (DType::I4, &OpAttrs::None)
        | (DType::I2, &OpAttrs::None)
        | (DType::I1, &OpAttrs::None)
        | (DType::U4, &OpAttrs::None)
        | (DType::U2, &OpAttrs::None)
        | (DType::U1, &OpAttrs::None)
        | (DType::Bool, &OpAttrs::None)
        | (DType::Bitset, &OpAttrs::None)
        | (DType::F32, &OpAttrs::None)
        | (DType::F64, &OpAttrs::None) => {
            Ok(format!("matmul_{}", super::dtype_suffix(dtype).unwrap()))
        }
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for matmul dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}

pub(crate) fn spv_target_name_matmul_accumulate(
    input_dtype: DType,
    output_dtype: DType,
    attrs: &OpAttrs,
) -> Result<String> {
    match (input_dtype, output_dtype, attrs) {
        (DType::I8, DType::I16, &OpAttrs::Accumulate { dtype: DType::I16 })
        | (DType::I8, DType::I32, &OpAttrs::Accumulate { dtype: DType::I32 })
        | (DType::I8, DType::I64, &OpAttrs::Accumulate { dtype: DType::I64 })
        | (DType::I16, DType::I32, &OpAttrs::Accumulate { dtype: DType::I32 })
        | (DType::I16, DType::I64, &OpAttrs::Accumulate { dtype: DType::I64 })
        | (DType::I32, DType::I64, &OpAttrs::Accumulate { dtype: DType::I64 })
        | (DType::U8, DType::U16, &OpAttrs::Accumulate { dtype: DType::U16 })
        | (DType::U8, DType::U32, &OpAttrs::Accumulate { dtype: DType::U32 })
        | (DType::U8, DType::U64, &OpAttrs::Accumulate { dtype: DType::U64 })
        | (DType::U16, DType::U32, &OpAttrs::Accumulate { dtype: DType::U32 })
        | (DType::U16, DType::U64, &OpAttrs::Accumulate { dtype: DType::U64 })
        | (DType::U32, DType::U64, &OpAttrs::Accumulate { dtype: DType::U64 })
        | (DType::I4, DType::I8, &OpAttrs::Accumulate { dtype: DType::I8 })
        | (DType::I4, DType::I16, &OpAttrs::Accumulate { dtype: DType::I16 })
        | (DType::I4, DType::I32, &OpAttrs::Accumulate { dtype: DType::I32 })
        | (DType::I4, DType::I64, &OpAttrs::Accumulate { dtype: DType::I64 })
        | (DType::I2, DType::I8, &OpAttrs::Accumulate { dtype: DType::I8 })
        | (DType::I2, DType::I16, &OpAttrs::Accumulate { dtype: DType::I16 })
        | (DType::I2, DType::I32, &OpAttrs::Accumulate { dtype: DType::I32 })
        | (DType::I2, DType::I64, &OpAttrs::Accumulate { dtype: DType::I64 })
        | (DType::I1, DType::I8, &OpAttrs::Accumulate { dtype: DType::I8 })
        | (DType::I1, DType::I16, &OpAttrs::Accumulate { dtype: DType::I16 })
        | (DType::I1, DType::I32, &OpAttrs::Accumulate { dtype: DType::I32 })
        | (DType::I1, DType::I64, &OpAttrs::Accumulate { dtype: DType::I64 })
        | (DType::U4, DType::U8, &OpAttrs::Accumulate { dtype: DType::U8 })
        | (DType::U4, DType::U16, &OpAttrs::Accumulate { dtype: DType::U16 })
        | (DType::U4, DType::U32, &OpAttrs::Accumulate { dtype: DType::U32 })
        | (DType::U4, DType::U64, &OpAttrs::Accumulate { dtype: DType::U64 })
        | (DType::U2, DType::U8, &OpAttrs::Accumulate { dtype: DType::U8 })
        | (DType::U2, DType::U16, &OpAttrs::Accumulate { dtype: DType::U16 })
        | (DType::U2, DType::U32, &OpAttrs::Accumulate { dtype: DType::U32 })
        | (DType::U2, DType::U64, &OpAttrs::Accumulate { dtype: DType::U64 })
        | (DType::U1, DType::U8, &OpAttrs::Accumulate { dtype: DType::U8 })
        | (DType::U1, DType::U16, &OpAttrs::Accumulate { dtype: DType::U16 })
        | (DType::U1, DType::U32, &OpAttrs::Accumulate { dtype: DType::U32 })
        | (DType::U1, DType::U64, &OpAttrs::Accumulate { dtype: DType::U64 }) => Ok(format!(
            "matmul_{}_{}",
            super::dtype_suffix(input_dtype).unwrap(),
            super::dtype_suffix(output_dtype).unwrap()
        )),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for matmul accumulate input {:?}, output {:?}, attrs {:?}",
            input_dtype,
            output_dtype,
            attrs
        )),
    }
}
