use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes_for_len;
use crate::backend::VulkanBuffer;
use crate::graph::{OpAttrs, OpKind};
use crate::tensor::{broadcast_strides, compute_strides, numel, DType};
use crate::timer::Timer;

pub mod registry;
pub mod registry_inplace;

fn matmul_dims(
    a: &VulkanBuffer,
    b: &VulkanBuffer,
) -> Result<(Vec<usize>, usize, usize, usize, Vec<usize>)> {
    if a.shape.len() < 2 || b.shape.len() < 2 {
        return Err(anyhow!(
            "matmul expects >=2D inputs, got {:?} and {:?}",
            a.shape,
            b.shape
        ));
    }
    let rank = a.shape.len().max(b.shape.len());
    let mut a_aligned = vec![1; rank - a.shape.len()];
    a_aligned.extend_from_slice(&a.shape);
    let mut b_aligned = vec![1; rank - b.shape.len()];
    b_aligned.extend_from_slice(&b.shape);
    let m = a_aligned[rank - 2];
    let k = a_aligned[rank - 1];
    let k2 = b_aligned[rank - 2];
    let n = b_aligned[rank - 1];
    if k != k2 {
        return Err(anyhow!(
            "matmul inner dims must match, got {:?} and {:?}",
            a.shape,
            b.shape
        ));
    }
    let batch_shape = crate::tensor::broadcast_shapes(&a_aligned[..rank - 2], &b_aligned[..rank - 2])?;
    let mut out_shape = batch_shape.clone();
    out_shape.push(m);
    out_shape.push(n);
    Ok((batch_shape, m, k, n, out_shape))
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
    let (batch_shape, m, k, n, out_shape) = matmul_dims(a, b)?;
    let batch = numel(&batch_shape);
    let runtime = super::runtime_from_buffers(a, Some(b))?;
    let target = if a.effective_dtype == DType::F16 && runtime.supports_f16() {
        "matmul_f16_native".to_string()
    } else {
        super::spv_target_name(OpKind::Matmul, a.effective_dtype, attrs)?
    };
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for matmul op", target))?;
    let len = m.saturating_mul(n);
    let total_len = batch.saturating_mul(len);
    let output_size = storage_size_bytes_for_len(a.effective_dtype, total_len);
    let output_inner = runtime.create_buffer(output_size)?;
    let push = [len as u32, m as u32, n as u32, k as u32];
    let bits = a.effective_dtype.bit_width() as usize;
    let packed = a.effective_dtype.is_packed();
    let a_batch_elems = m.saturating_mul(k);
    let b_batch_elems = k.saturating_mul(n);
    let out_batch_elems = len;
    let (_a_batch_bytes, _b_batch_bytes, out_batch_bytes) = if packed {
        let a_bits = a_batch_elems.saturating_mul(bits);
        let b_bits = b_batch_elems.saturating_mul(bits);
        let out_bits = out_batch_elems.saturating_mul(bits);
        if a_bits % 8 != 0 || b_bits % 8 != 0 || out_bits % 8 != 0 {
            return Err(anyhow!(
                "vulkan batched matmul requires byte-aligned packed batches (dtype {:?})",
                a.effective_dtype
            ));
        }
        (a_bits / 8, b_bits / 8, out_bits / 8)
    } else {
        let elem_bytes = storage_size_bytes_for_len(a.effective_dtype, 1);
        (
            a_batch_elems.saturating_mul(elem_bytes),
            b_batch_elems.saturating_mul(elem_bytes),
            out_batch_elems.saturating_mul(elem_bytes),
        )
    };
    let elem_bytes = storage_size_bytes_for_len(a.effective_dtype, 1);
    let rank = batch_shape.len() + 2;
    let mut a_aligned = vec![1; rank - a.shape.len()];
    a_aligned.extend_from_slice(&a.shape);
    let mut b_aligned = vec![1; rank - b.shape.len()];
    b_aligned.extend_from_slice(&b.shape);
    let mut a_strides = vec![0usize; rank - a.strides.len()];
    a_strides.extend_from_slice(&a.strides);
    let mut b_strides = vec![0usize; rank - b.strides.len()];
    b_strides.extend_from_slice(&b.strides);
    let a_batch_strides = broadcast_strides(
        &a_aligned[..rank - 2],
        &a_strides[..rank - 2],
        &batch_shape,
    )?;
    let b_batch_strides = broadcast_strides(
        &b_aligned[..rank - 2],
        &b_strides[..rank - 2],
        &batch_shape,
    )?;
    let mut batch_coords = vec![0usize; batch_shape.len()];
    let mut total_duration = 0u128;
    for batch_idx in 0..batch {
        let mut a_batch_offset = 0usize;
        let mut b_batch_offset = 0usize;
        for (dim, coord) in batch_coords.iter().enumerate() {
            a_batch_offset =
                a_batch_offset.saturating_add(coord.saturating_mul(a_batch_strides[dim]));
            b_batch_offset =
                b_batch_offset.saturating_add(coord.saturating_mul(b_batch_strides[dim]));
        }
        let offsets = if packed {
            let a_bits_offset = a_batch_offset.saturating_mul(bits);
            let b_bits_offset = b_batch_offset.saturating_mul(bits);
            if a_bits_offset % 8 != 0 || b_bits_offset % 8 != 0 {
                return Err(anyhow!(
                    "vulkan matmul packed broadcast requires byte-aligned batch offsets (dtype {:?})",
                    a.effective_dtype
                ));
            }
            [
                (a_bits_offset / 8) as u64,
                (b_bits_offset / 8) as u64,
                (out_batch_bytes.saturating_mul(batch_idx)) as u64,
            ]
        } else {
            [
                (a_batch_offset.saturating_mul(elem_bytes)) as u64,
                (b_batch_offset.saturating_mul(elem_bytes)) as u64,
                (out_batch_bytes.saturating_mul(batch_idx)) as u64,
            ]
        };
        let duration_ns = runtime.dispatch_with_offsets(
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
            offsets,
        )?;
        total_duration = total_duration.saturating_add(duration_ns);
        for dim in (0..batch_shape.len()).rev() {
            batch_coords[dim] += 1;
            if batch_coords[dim] < batch_shape[dim] {
                break;
            }
            batch_coords[dim] = 0;
        }
    }
    Timer::record(thread_id, total_duration);
    let strides = compute_strides(out_shape.as_slice());
    Ok(VulkanBuffer {
        dtype: a.dtype,
        effective_dtype: a.effective_dtype,
        len: total_len,
        shape: out_shape,
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
    output: Option<&VulkanBuffer>,
    thread_id: usize,
) -> Result<VulkanBuffer> {
    if a.effective_dtype != b.effective_dtype {
        return Err(anyhow!("matmul accumulate expects matching dtypes"));
    }
    let (batch_shape, m, k, n, out_shape) = matmul_dims(a, b)?;
    let batch = numel(&batch_shape);
    let runtime = super::runtime_from_buffers(a, Some(b))?;
    let target = spv_target_name_matmul_accumulate(a.effective_dtype, output_dtype, attrs)?;
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for matmul accumulate", target))?;
    let len = m.saturating_mul(n);
    let total_len = batch.saturating_mul(len);
    let output_size = storage_size_bytes_for_len(output_dtype, total_len);
    let output_inner = match output {
        Some(out)
            if out.dtype == output_dtype
                && out.effective_dtype == output_dtype
                && out.len == total_len
                && (out.inner.size as usize) >= output_size =>
        {
            out.inner.clone()
        }
        _ => runtime.create_buffer(output_size)?,
    };
    let push = [len as u32, m as u32, n as u32, k as u32];
    let bits = a.effective_dtype.bit_width() as usize;
    let packed = a.effective_dtype.is_packed();
    let a_batch_elems = m.saturating_mul(k);
    let b_batch_elems = k.saturating_mul(n);
    let out_batch_elems = len;
    let (_a_batch_bytes, _b_batch_bytes, out_batch_bytes) = if packed {
        let a_bits = a_batch_elems.saturating_mul(bits);
        let b_bits = b_batch_elems.saturating_mul(bits);
        let out_bits = out_batch_elems.saturating_mul(output_dtype.bit_width() as usize);
        if a_bits % 8 != 0 || b_bits % 8 != 0 || out_bits % 8 != 0 {
            return Err(anyhow!(
                "vulkan batched matmul requires byte-aligned packed batches (dtype {:?})",
                a.effective_dtype
            ));
        }
        (a_bits / 8, b_bits / 8, out_bits / 8)
    } else {
        let a_elem_bytes = storage_size_bytes_for_len(a.effective_dtype, 1);
        let b_elem_bytes = storage_size_bytes_for_len(b.effective_dtype, 1);
        let out_elem_bytes = storage_size_bytes_for_len(output_dtype, 1);
        (
            a_batch_elems.saturating_mul(a_elem_bytes),
            b_batch_elems.saturating_mul(b_elem_bytes),
            out_batch_elems.saturating_mul(out_elem_bytes),
        )
    };
    let a_elem_bytes = storage_size_bytes_for_len(a.effective_dtype, 1);
    let b_elem_bytes = storage_size_bytes_for_len(b.effective_dtype, 1);
    let rank = batch_shape.len() + 2;
    let mut a_aligned = vec![1; rank - a.shape.len()];
    a_aligned.extend_from_slice(&a.shape);
    let mut b_aligned = vec![1; rank - b.shape.len()];
    b_aligned.extend_from_slice(&b.shape);
    let mut a_strides = vec![0usize; rank - a.strides.len()];
    a_strides.extend_from_slice(&a.strides);
    let mut b_strides = vec![0usize; rank - b.strides.len()];
    b_strides.extend_from_slice(&b.strides);
    let a_batch_strides = broadcast_strides(
        &a_aligned[..rank - 2],
        &a_strides[..rank - 2],
        &batch_shape,
    )?;
    let b_batch_strides = broadcast_strides(
        &b_aligned[..rank - 2],
        &b_strides[..rank - 2],
        &batch_shape,
    )?;
    let mut batch_coords = vec![0usize; batch_shape.len()];
    let mut total_duration = 0u128;
    for batch_idx in 0..batch {
        let mut a_batch_offset = 0usize;
        let mut b_batch_offset = 0usize;
        for (dim, coord) in batch_coords.iter().enumerate() {
            a_batch_offset =
                a_batch_offset.saturating_add(coord.saturating_mul(a_batch_strides[dim]));
            b_batch_offset =
                b_batch_offset.saturating_add(coord.saturating_mul(b_batch_strides[dim]));
        }
        let offsets = if packed {
            let a_bits_offset = a_batch_offset.saturating_mul(bits);
            let b_bits_offset = b_batch_offset.saturating_mul(bits);
            if a_bits_offset % 8 != 0 || b_bits_offset % 8 != 0 {
                return Err(anyhow!(
                    "vulkan matmul packed broadcast requires byte-aligned batch offsets (dtype {:?})",
                    a.effective_dtype
                ));
            }
            [
                (a_bits_offset / 8) as u64,
                (b_bits_offset / 8) as u64,
                (out_batch_bytes.saturating_mul(batch_idx)) as u64,
            ]
        } else {
            [
                (a_batch_offset.saturating_mul(a_elem_bytes)) as u64,
                (b_batch_offset.saturating_mul(b_elem_bytes)) as u64,
                (out_batch_bytes.saturating_mul(batch_idx)) as u64,
            ]
        };
        let duration_ns = runtime.dispatch_with_offsets(
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
            offsets,
        )?;
        total_duration = total_duration.saturating_add(duration_ns);
        for dim in (0..batch_shape.len()).rev() {
            batch_coords[dim] += 1;
            if batch_coords[dim] < batch_shape[dim] {
                break;
            }
            batch_coords[dim] = 0;
        }
    }
    Timer::record(thread_id, total_duration);
    let strides = compute_strides(out_shape.as_slice());
    Ok(VulkanBuffer {
        dtype: output_dtype,
        effective_dtype: output_dtype,
        len: total_len,
        shape: out_shape,
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
        | (DType::F16, &OpAttrs::None)
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
