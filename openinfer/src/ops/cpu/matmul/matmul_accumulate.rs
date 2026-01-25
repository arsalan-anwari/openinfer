use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::is_contiguous;
use crate::tensor::{numel, Tensor};
use crate::timer::Timer;

use super::matmul::matmul_dims;

fn ensure_output_len<T>(out: &Tensor<T>, len: usize) -> Result<()> {
    if out.data.len() != len {
        return Err(anyhow!("matmul output shape mismatch"));
    }
    Ok(())
}

fn ensure_matmul_accumulate_layout<T, O>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &Tensor<O>,
    batch_shape: &[usize],
    m: usize,
    n: usize,
) -> Result<()> {
    let mut expected_shape = batch_shape.to_vec();
    expected_shape.push(m);
    expected_shape.push(n);
    if out.shape() != expected_shape.as_slice() {
        return Err(anyhow!(
            "matmul output shape mismatch: expected {:?}, got {:?}",
            expected_shape,
            out.shape()
        ));
    }
    if !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("matmul output must be contiguous"));
    }
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(b.shape(), b.strides()) {
        return Err(anyhow!("matmul inputs must be contiguous"));
    }
    if a.data.len() != numel(a.shape()) || b.data.len() != numel(b.shape()) {
        return Err(anyhow!("matmul input length mismatch"));
    }
    Ok(())
}

macro_rules! matmul_accumulate_signed {
    ($name:ident, $in:ty, $out:ty) => {
        pub fn $name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            out: &mut Tensor<$out>,
            thread_id: usize,
        ) -> Result<()> {
            let (batch_shape, m, k, n) = matmul_dims(a.shape(), b.shape())?;
            let batch = numel(&batch_shape);
            let len = batch * m * n;
            ensure_output_len(out, len)?;
            ensure_matmul_accumulate_layout(a, b, out, &batch_shape, m, n)?;
            Timer::start(thread_id);
            let rank = batch_shape.len() + 2;
            let mut a_aligned = vec![1; rank - a.shape().len()];
            a_aligned.extend_from_slice(a.shape());
            let mut b_aligned = vec![1; rank - b.shape().len()];
            b_aligned.extend_from_slice(b.shape());
            let mut a_strides = vec![0; rank - a.strides().len()];
            a_strides.extend_from_slice(a.strides());
            let mut b_strides = vec![0; rank - b.strides().len()];
            b_strides.extend_from_slice(b.strides());
            let a_batch_strides = crate::tensor::broadcast_strides(
                &a_aligned[..rank - 2],
                &a_strides[..rank - 2],
                &batch_shape,
            )?;
            let b_batch_strides = crate::tensor::broadcast_strides(
                &b_aligned[..rank - 2],
                &b_strides[..rank - 2],
                &batch_shape,
            )?;
            let a_stride_m = a_strides[rank - 2];
            let a_stride_k = a_strides[rank - 1];
            let b_stride_k = b_strides[rank - 2];
            let b_stride_n = b_strides[rank - 1];
            let mut batch_coords = vec![0usize; batch_shape.len()];
            for batch_idx in 0..batch {
                let mut a_batch_offset = 0usize;
                let mut b_batch_offset = 0usize;
                for (dim, coord) in batch_coords.iter().enumerate() {
                    a_batch_offset =
                        a_batch_offset.saturating_add(coord.saturating_mul(a_batch_strides[dim]));
                    b_batch_offset =
                        b_batch_offset.saturating_add(coord.saturating_mul(b_batch_strides[dim]));
                }
                let out_base = batch_idx * m * n;
                for i in 0..m {
                    let a_row_base = a_batch_offset + i * a_stride_m;
                    for j in 0..n {
                        let mut acc: $out = 0;
                        for kk in 0..k {
                            acc = acc.wrapping_add(
                                (a.data[a_row_base + kk * a_stride_k] as $out)
                                    .wrapping_mul(
                                        b.data[b_batch_offset + kk * b_stride_k + j * b_stride_n]
                                            as $out,
                                    ),
                            );
                        }
                        out.data[out_base + i * n + j] = acc;
                    }
                }
                for dim in (0..batch_shape.len()).rev() {
                    batch_coords[dim] += 1;
                    if batch_coords[dim] < batch_shape[dim] {
                        break;
                    }
                    batch_coords[dim] = 0;
                }
            }
            Timer::stop(thread_id);
            Ok(())
        }
    };
}

macro_rules! matmul_accumulate_unsigned {
    ($name:ident, $in:ty, $out:ty) => {
        pub fn $name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            out: &mut Tensor<$out>,
            thread_id: usize,
        ) -> Result<()> {
            let (batch_shape, m, k, n) = matmul_dims(a.shape(), b.shape())?;
            let batch = numel(&batch_shape);
            let len = batch * m * n;
            ensure_output_len(out, len)?;
            ensure_matmul_accumulate_layout(a, b, out, &batch_shape, m, n)?;
            Timer::start(thread_id);
            let rank = batch_shape.len() + 2;
            let mut a_aligned = vec![1; rank - a.shape().len()];
            a_aligned.extend_from_slice(a.shape());
            let mut b_aligned = vec![1; rank - b.shape().len()];
            b_aligned.extend_from_slice(b.shape());
            let mut a_strides = vec![0; rank - a.strides().len()];
            a_strides.extend_from_slice(a.strides());
            let mut b_strides = vec![0; rank - b.strides().len()];
            b_strides.extend_from_slice(b.strides());
            let a_batch_strides = crate::tensor::broadcast_strides(
                &a_aligned[..rank - 2],
                &a_strides[..rank - 2],
                &batch_shape,
            )?;
            let b_batch_strides = crate::tensor::broadcast_strides(
                &b_aligned[..rank - 2],
                &b_strides[..rank - 2],
                &batch_shape,
            )?;
            let a_stride_m = a_strides[rank - 2];
            let a_stride_k = a_strides[rank - 1];
            let b_stride_k = b_strides[rank - 2];
            let b_stride_n = b_strides[rank - 1];
            let mut batch_coords = vec![0usize; batch_shape.len()];
            for batch_idx in 0..batch {
                let mut a_batch_offset = 0usize;
                let mut b_batch_offset = 0usize;
                for (dim, coord) in batch_coords.iter().enumerate() {
                    a_batch_offset =
                        a_batch_offset.saturating_add(coord.saturating_mul(a_batch_strides[dim]));
                    b_batch_offset =
                        b_batch_offset.saturating_add(coord.saturating_mul(b_batch_strides[dim]));
                }
                let out_base = batch_idx * m * n;
                for i in 0..m {
                    let a_row_base = a_batch_offset + i * a_stride_m;
                    for j in 0..n {
                        let mut acc: $out = 0;
                        for kk in 0..k {
                            acc = acc.wrapping_add(
                                (a.data[a_row_base + kk * a_stride_k] as $out)
                                    .wrapping_mul(
                                        b.data[b_batch_offset + kk * b_stride_k + j * b_stride_n]
                                            as $out,
                                    ),
                            );
                        }
                        out.data[out_base + i * n + j] = acc;
                    }
                }
                for dim in (0..batch_shape.len()).rev() {
                    batch_coords[dim] += 1;
                    if batch_coords[dim] < batch_shape[dim] {
                        break;
                    }
                    batch_coords[dim] = 0;
                }
            }
            Timer::stop(thread_id);
            Ok(())
        }
    };
}

matmul_accumulate_signed!(matmul_i8_i16, i8, i16);
matmul_accumulate_signed!(matmul_i8_i32, i8, i32);
matmul_accumulate_signed!(matmul_i8_i64, i8, i64);
matmul_accumulate_signed!(matmul_i16_i32, i16, i32);
matmul_accumulate_signed!(matmul_i16_i64, i16, i64);
matmul_accumulate_signed!(matmul_i32_i64, i32, i64);
matmul_accumulate_unsigned!(matmul_u8_u16, u8, u16);
matmul_accumulate_unsigned!(matmul_u8_u32, u8, u32);
matmul_accumulate_unsigned!(matmul_u8_u64, u8, u64);
matmul_accumulate_unsigned!(matmul_u16_u32, u16, u32);
matmul_accumulate_unsigned!(matmul_u16_u64, u16, u64);
matmul_accumulate_unsigned!(matmul_u32_u64, u32, u64);
