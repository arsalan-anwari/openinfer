use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes_for_len;
use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::graph::OpKind;
use crate::tensor::{compute_strides, DType};
use crate::timer::Timer;

pub mod registry;
pub mod registry_accumulate;
pub mod registry_inplace;

pub fn abs_generic(attrs: &OpAttrs, a: &VulkanBuffer, thread_id: usize) -> Result<VulkanBuffer> {
    let runtime = super::runtime_from_buffers(a, None)?;
    let target = if a.effective_dtype == DType::F16 && runtime.use_native_f16() {
        "abs_f16_native".to_string()
    } else {
        super::spv_target_name(OpKind::Abs, a.effective_dtype, attrs)?
    };
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for abs op", target))?;
    let output_size = storage_size_bytes_for_len(a.effective_dtype, a.len);
    let output_inner = runtime.create_buffer(output_size)?;
    let push = [a.len as u32, 0, 0, 0];
    let duration_ns = runtime.dispatch(
        OpKind::Abs,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &a.inner,
        &output_inner,
        &push,
        a.len,
    )?;
    Timer::record(thread_id, duration_ns);
    Ok(VulkanBuffer {
        dtype: a.dtype,
        effective_dtype: a.effective_dtype,
        len: a.len,
        shape: a.shape.clone(),
        strides: compute_strides(a.shape.as_slice()),
        shader: a.shader.clone(),
        inner: output_inner,
    })
}

pub fn abs_accumulate_generic(
    attrs: &OpAttrs,
    a: &VulkanBuffer,
    output_dtype: DType,
    output: Option<&VulkanBuffer>,
    thread_id: usize,
) -> Result<VulkanBuffer> {
    let runtime = super::runtime_from_buffers(a, None)?;
    let target = spv_target_name_abs_accumulate(a.effective_dtype, output_dtype, attrs)?;
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for abs accumulate", target))?;
    let output_size = storage_size_bytes_for_len(output_dtype, a.len);
    let output_inner = match output {
        Some(out)
            if out.dtype == output_dtype
                && out.effective_dtype == output_dtype
                && out.len == a.len
                && (out.inner.size as usize) >= output_size =>
        {
            out.inner.clone()
        }
        _ => runtime.create_buffer(output_size)?,
    };
    let push = [a.len as u32, 0, 0, 0];
    let duration_ns = runtime.dispatch(
        OpKind::Abs,
        output_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &a.inner,
        &output_inner,
        &push,
        a.len,
    )?;
    Timer::record(thread_id, duration_ns);
    Ok(VulkanBuffer {
        dtype: output_dtype,
        effective_dtype: output_dtype,
        len: a.len,
        shape: a.shape.clone(),
        strides: compute_strides(a.shape.as_slice()),
        shader: a.shader.clone(),
        inner: output_inner,
    })
}

pub fn abs_inplace_generic(attrs: &OpAttrs, a: &VulkanBuffer, thread_id: usize) -> Result<VulkanBuffer> {
    let runtime = super::runtime_from_buffers(a, None)?;
    let target = if a.effective_dtype == DType::F16 && runtime.use_native_f16() {
        "abs_inplace_f16_native".to_string()
    } else {
        spv_target_name_abs_inplace(a.effective_dtype, attrs)?
    };
    let entry = "main";
    let output_size = storage_size_bytes_for_len(a.effective_dtype, a.len);
    if output_size > a.inner.size as usize {
        return Err(anyhow!("abs inplace output buffer too small"));
    }
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for abs inplace", target))?;
    let push = [a.len as u32, 0, 0, 0];
    let duration_ns = runtime.dispatch(
        OpKind::Abs,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &a.inner,
        &a.inner,
        &push,
        a.len,
    )?;
    Timer::record(thread_id, duration_ns);
    Ok(VulkanBuffer {
        dtype: a.dtype,
        effective_dtype: a.effective_dtype,
        len: a.len,
        shape: a.shape.clone(),
        strides: compute_strides(a.shape.as_slice()),
        shader: a.shader.clone(),
        inner: a.inner.clone(),
    })
}

pub(crate) fn spv_target_name_abs(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::I8, &OpAttrs::None)
        | (DType::I16, &OpAttrs::None)
        | (DType::I32, &OpAttrs::None)
        | (DType::I64, &OpAttrs::None)
        | (DType::I4, &OpAttrs::None)
        | (DType::I2, &OpAttrs::None)
        | (DType::I1, &OpAttrs::None)
        | (DType::F16, &OpAttrs::None)
        | (DType::BF16, &OpAttrs::None)
        | (DType::F8E5M2, &OpAttrs::None)
        | (DType::F32, &OpAttrs::None)
        | (DType::F64, &OpAttrs::None) => Ok(format!("abs_{}", super::dtype_suffix(dtype).unwrap())),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for abs dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}

pub(crate) fn spv_target_name_abs_accumulate(
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
        | (DType::I1, DType::I64, &OpAttrs::Accumulate { dtype: DType::I64 }) => Ok(format!(
            "abs_{}_{}",
            super::dtype_suffix(input_dtype).unwrap(),
            super::dtype_suffix(output_dtype).unwrap()
        )),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for abs accumulate input {:?}, output {:?}, attrs {:?}",
            input_dtype,
            output_dtype,
            attrs
        )),
    }
}

pub(crate) fn spv_target_name_abs_inplace(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::I8, &OpAttrs::None)
        | (DType::I16, &OpAttrs::None)
        | (DType::I32, &OpAttrs::None)
        | (DType::I64, &OpAttrs::None)
        | (DType::I4, &OpAttrs::None)
        | (DType::I2, &OpAttrs::None)
        | (DType::I1, &OpAttrs::None)
        | (DType::F16, &OpAttrs::None)
        | (DType::BF16, &OpAttrs::None)
        | (DType::F8E5M2, &OpAttrs::None)
        | (DType::F32, &OpAttrs::None)
        | (DType::F64, &OpAttrs::None) => Ok(format!(
            "abs_inplace_{}",
            super::dtype_suffix(dtype).unwrap()
        )),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for abs inplace dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}
