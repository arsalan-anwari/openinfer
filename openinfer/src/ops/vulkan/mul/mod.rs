use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes_for_len;
use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::graph::OpKind;
use crate::tensor::{broadcast_shapes, compute_strides, numel, DType};
use crate::timer::Timer;

pub mod registry;
pub mod registry_inplace;

pub fn mul_generic(attrs: &OpAttrs, a: &VulkanBuffer, b: &VulkanBuffer, thread_id: usize) -> Result<VulkanBuffer> {
    let allow_mixed = a.shader_setting_bool("allow_mixed_dtypes").unwrap_or(false);
    if !allow_mixed && a.effective_dtype != b.effective_dtype {
        return Err(anyhow!("mul op expects matching dtypes"));
    }
    let out_shape = if a.shape == b.shape {
        a.shape.clone()
    } else {
        broadcast_shapes(&a.shape, &b.shape)?
    };
    let len = numel(&out_shape);
    let runtime = super::runtime_from_buffers(a, Some(b))?;
    let target = if a.effective_dtype == DType::F16 && runtime.supports_f16() {
        "mul_f16_native".to_string()
    } else {
        super::spv_target_name(OpKind::Mul, a.effective_dtype, attrs)?
    };
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for mul op", target))?;
    let output_size = storage_size_bytes_for_len(a.effective_dtype, len);
    let output_inner = runtime.create_buffer(output_size)?;
    let meta_a = crate::backend::vulkan::broadcast::build_metadata(
        a.shape.as_slice(),
        a.strides.as_slice(),
        out_shape.as_slice(),
    )?;
    let meta_b = crate::backend::vulkan::broadcast::build_metadata(
        b.shape.as_slice(),
        b.strides.as_slice(),
        out_shape.as_slice(),
    )?;
    let meta_a_bytes: Vec<u8> = meta_a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let meta_b_bytes: Vec<u8> = meta_b.iter().flat_map(|v| v.to_le_bytes()).collect();
    let meta_a_inner = runtime.create_buffer(meta_a_bytes.len())?;
    let meta_b_inner = runtime.create_buffer(meta_b_bytes.len())?;
    runtime.write_buffer(&meta_a_inner, &meta_a_bytes)?;
    runtime.write_buffer(&meta_b_inner, &meta_b_bytes)?;
    let push = [len as u32, 0, 0, 0];
    let duration_ns = runtime.dispatch_with_offsets_meta(
        OpKind::Mul,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &b.inner,
        &output_inner,
        &meta_a_inner,
        &meta_b_inner,
        push,
        len,
        [0, 0, 0],
    )?;
    Timer::record(thread_id, duration_ns);
    let strides = compute_strides(out_shape.as_slice());
    Ok(VulkanBuffer {
        dtype: a.dtype,
        effective_dtype: a.effective_dtype,
        len,
        shape: out_shape,
        strides,
        shader: a.shader.clone(),
        inner: output_inner,
    })
}

pub fn mul_accumulate_generic(
    attrs: &OpAttrs,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    output_dtype: DType,
    output: Option<&VulkanBuffer>,
    thread_id: usize,
) -> Result<VulkanBuffer> {
    if a.effective_dtype != b.effective_dtype {
        return Err(anyhow!("mul accumulate expects matching input dtypes"));
    }
    let out_shape = if a.shape == b.shape {
        a.shape.clone()
    } else {
        broadcast_shapes(&a.shape, &b.shape)?
    };
    let len = numel(&out_shape);
    let runtime = super::runtime_from_buffers(a, Some(b))?;
    let target = spv_target_name_mul_accumulate(a.effective_dtype, output_dtype, attrs)?;
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for mul accumulate", target))?;
    let output_size = storage_size_bytes_for_len(output_dtype, len);
    let output_inner = match output {
        Some(out)
            if out.dtype == output_dtype
                && out.effective_dtype == output_dtype
                && out.len == len
                && (out.inner.size as usize) >= output_size =>
        {
            out.inner.clone()
        }
        _ => runtime.create_buffer(output_size)?,
    };
    let meta_a = crate::backend::vulkan::broadcast::build_metadata(
        a.shape.as_slice(),
        a.strides.as_slice(),
        out_shape.as_slice(),
    )?;
    let meta_b = crate::backend::vulkan::broadcast::build_metadata(
        b.shape.as_slice(),
        b.strides.as_slice(),
        out_shape.as_slice(),
    )?;
    let meta_a_bytes: Vec<u8> = meta_a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let meta_b_bytes: Vec<u8> = meta_b.iter().flat_map(|v| v.to_le_bytes()).collect();
    let meta_a_inner = runtime.create_buffer(meta_a_bytes.len())?;
    let meta_b_inner = runtime.create_buffer(meta_b_bytes.len())?;
    runtime.write_buffer(&meta_a_inner, &meta_a_bytes)?;
    runtime.write_buffer(&meta_b_inner, &meta_b_bytes)?;
    let push = [len as u32, 0, 0, 0];
    let duration_ns = runtime.dispatch_with_offsets_meta(
        OpKind::Mul,
        output_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &b.inner,
        &output_inner,
        &meta_a_inner,
        &meta_b_inner,
        push,
        len,
        [0, 0, 0],
    )?;
    Timer::record(thread_id, duration_ns);
    let strides = compute_strides(out_shape.as_slice());
    Ok(VulkanBuffer {
        dtype: output_dtype,
        effective_dtype: output_dtype,
        len,
        shape: out_shape,
        strides,
        shader: a.shader.clone(),
        inner: output_inner,
    })
}

pub fn mul_inplace_generic(
    attrs: &OpAttrs,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    thread_id: usize,
) -> Result<VulkanBuffer> {
    let allow_mixed = a.shader_setting_bool("allow_mixed_dtypes").unwrap_or(false);
    if !allow_mixed && a.effective_dtype != b.effective_dtype {
        return Err(anyhow!("mul inplace expects matching dtypes"));
    }
    let out_shape = if a.shape == b.shape {
        a.shape.clone()
    } else {
        broadcast_shapes(&a.shape, &b.shape)?
    };
    let len = numel(&out_shape);
    let runtime = super::runtime_from_buffers(a, Some(b))?;
    let target = if a.effective_dtype == DType::F16 && runtime.supports_f16() {
        "mul_inplace_f16_native".to_string()
    } else {
        spv_target_name_mul_inplace(a.effective_dtype, attrs)?
    };
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for mul inplace", target))?;
    let output_size = storage_size_bytes_for_len(a.effective_dtype, len);
    if output_size > a.inner.size as usize {
        return Err(anyhow!("mul inplace output buffer too small"));
    }
    let meta_a = crate::backend::vulkan::broadcast::build_metadata(
        a.shape.as_slice(),
        a.strides.as_slice(),
        out_shape.as_slice(),
    )?;
    let meta_b = crate::backend::vulkan::broadcast::build_metadata(
        b.shape.as_slice(),
        b.strides.as_slice(),
        out_shape.as_slice(),
    )?;
    let meta_a_bytes: Vec<u8> = meta_a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let meta_b_bytes: Vec<u8> = meta_b.iter().flat_map(|v| v.to_le_bytes()).collect();
    let meta_a_inner = runtime.create_buffer(meta_a_bytes.len())?;
    let meta_b_inner = runtime.create_buffer(meta_b_bytes.len())?;
    runtime.write_buffer(&meta_a_inner, &meta_a_bytes)?;
    runtime.write_buffer(&meta_b_inner, &meta_b_bytes)?;
    let push = [len as u32, 0, 0, 0];
    let duration_ns = runtime.dispatch_with_offsets_meta(
        OpKind::Mul,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &b.inner,
        &a.inner,
        &meta_a_inner,
        &meta_b_inner,
        push,
        len,
        [0, 0, 0],
    )?;
    Timer::record(thread_id, duration_ns);
    Ok(VulkanBuffer {
        dtype: a.dtype,
        effective_dtype: a.effective_dtype,
        len,
        shape: out_shape.clone(),
        strides: compute_strides(out_shape.as_slice()),
        shader: a.shader.clone(),
        inner: a.inner.clone(),
    })
}

pub(crate) fn spv_target_name_mul(dtype: DType, attrs: &OpAttrs) -> Result<String> {
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
        | (DType::F16, &OpAttrs::None)
        | (DType::BF16, &OpAttrs::None)
        | (DType::F8E5M2, &OpAttrs::None)
        | (DType::F32, &OpAttrs::None)
        | (DType::F64, &OpAttrs::None) => Ok(format!("mul_{}", super::dtype_suffix(dtype).unwrap())),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for mul dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}

pub(crate) fn spv_target_name_mul_accumulate(
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
            "mul_{}_{}",
            super::dtype_suffix(input_dtype).unwrap(),
            super::dtype_suffix(output_dtype).unwrap()
        )),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for mul accumulate input {:?}, output {:?}, attrs {:?}",
            input_dtype,
            output_dtype,
            attrs
        )),
    }
}

pub(crate) fn spv_target_name_mul_inplace(dtype: DType, attrs: &OpAttrs) -> Result<String> {
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
        | (DType::F16, &OpAttrs::None)
        | (DType::BF16, &OpAttrs::None)
        | (DType::F8E5M2, &OpAttrs::None)
        | (DType::F32, &OpAttrs::None)
        | (DType::F64, &OpAttrs::None) => Ok(format!(
            "mul_inplace_{}",
            super::dtype_suffix(dtype).unwrap()
        )),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for mul inplace dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}
