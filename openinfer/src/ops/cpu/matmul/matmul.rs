use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::is_contiguous;
use crate::tensor::{broadcast_shapes, numel, Bitset, F16, Tensor};
use crate::timer::Timer;

pub(crate) fn matmul_dims(
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<(Vec<usize>, usize, usize, usize)> {
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(anyhow!(
            "matmul expects >=2D inputs, got {:?} and {:?}",
            a_shape,
            b_shape
        ));
    }
    let rank = a_shape.len().max(b_shape.len());
    let mut a_aligned = vec![1; rank - a_shape.len()];
    a_aligned.extend_from_slice(a_shape);
    let mut b_aligned = vec![1; rank - b_shape.len()];
    b_aligned.extend_from_slice(b_shape);
    let m = a_aligned[rank - 2];
    let k = a_aligned[rank - 1];
    let k2 = b_aligned[rank - 2];
    let n = b_aligned[rank - 1];
    if k != k2 {
        return Err(anyhow!(
            "matmul inner dims must match, got {:?} and {:?}",
            a_shape,
            b_shape
        ));
    }
    let batch_shape = broadcast_shapes(&a_aligned[..rank - 2], &b_aligned[..rank - 2])?;
    Ok((batch_shape, m, k, n))
}

fn matmul_broadcast_into<T, Acc, F, G>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    thread_id: usize,
    zero: T,
    init: Acc,
    mut accumulate: F,
    mut finalize: G,
) -> Result<()>
where
    T: Clone,
    Acc: Copy,
    F: FnMut(Acc, &T, &T) -> Acc,
    G: FnMut(Acc) -> T,
{
    let (batch_shape, m, k, n) = matmul_dims(a.shape(), b.shape())?;
    let mut expected_shape = batch_shape.clone();
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
    let out_len = numel(out.shape());
    if out.data.len() != out_len {
        return Err(anyhow!("matmul output length mismatch"));
    }
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(b.shape(), b.strides()) {
        return Err(anyhow!("matmul inputs must be contiguous"));
    }
    let batch = numel(&batch_shape);
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
    out.data.fill(zero);
    let mut batch_coords = vec![0usize; batch_shape.len()];
    Timer::start(thread_id);
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
                let mut acc = init;
                for kk in 0..k {
                    let a_val = &a.data[a_row_base + kk * a_stride_k];
                    let b_val = &b.data[b_batch_offset + kk * b_stride_k + j * b_stride_n];
                    acc = accumulate(acc, a_val, b_val);
                }
                out.data[out_base + i * n + j] = finalize(acc);
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

pub fn matmul_f32(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>, thread_id: usize) -> Result<()> {
    matmul_broadcast_into(
        a,
        b,
        out,
        thread_id,
        0.0f32,
        0.0f32,
        |acc, x, y| acc + (*x * *y),
        |acc| acc,
    )
}

pub fn matmul_f64(a: &Tensor<f64>, b: &Tensor<f64>, out: &mut Tensor<f64>, thread_id: usize) -> Result<()> {
    matmul_broadcast_into(
        a,
        b,
        out,
        thread_id,
        0.0f64,
        0.0f64,
        |acc, x, y| acc + (*x * *y),
        |acc| acc,
    )
}

pub fn matmul_f16(a: &Tensor<F16>, b: &Tensor<F16>, out: &mut Tensor<F16>, thread_id: usize) -> Result<()> {
    matmul_broadcast_into(
        a,
        b,
        out,
        thread_id,
        F16::from_f32(0.0),
        0.0f32,
        |acc, x, y| acc + x.to_f32() * y.to_f32(),
        |acc| F16::from_f32(acc),
    )
}

pub fn matmul_bool(a: &Tensor<bool>, b: &Tensor<bool>, out: &mut Tensor<bool>, thread_id: usize) -> Result<()> {
    matmul_broadcast_into(
        a,
        b,
        out,
        thread_id,
        false,
        false,
        |acc, x, y| acc || (*x && *y),
        |acc| acc,
    )
}

pub fn matmul_bitset(
    a: &Tensor<Bitset>,
    b: &Tensor<Bitset>,
    out: &mut Tensor<Bitset>,
    thread_id: usize,
) -> Result<()> {
    matmul_broadcast_into(
        a,
        b,
        out,
        thread_id,
        Bitset { bits: 0 },
        0u64,
        |acc, x, y| acc.wrapping_add((x.bits as u64).wrapping_mul(y.bits as u64)),
        |acc| Bitset { bits: acc as u8 },
    )
}

macro_rules! matmul_signed {
    ($name:ident, $ty:ty, $acc:ty) => {
        pub fn $name(
            a: &Tensor<$ty>,
            b: &Tensor<$ty>,
            out: &mut Tensor<$ty>,
            thread_id: usize,
        ) -> Result<()> {
            matmul_broadcast_into(
                a,
                b,
                out,
                thread_id,
                0 as $ty,
                0 as $acc,
                |acc, x, y| acc.wrapping_add((*x as $acc).wrapping_mul(*y as $acc)),
                |acc| acc as $ty,
            )
        }
    };
}

macro_rules! matmul_unsigned {
    ($name:ident, $ty:ty, $acc:ty) => {
        pub fn $name(
            a: &Tensor<$ty>,
            b: &Tensor<$ty>,
            out: &mut Tensor<$ty>,
            thread_id: usize,
        ) -> Result<()> {
            matmul_broadcast_into(
                a,
                b,
                out,
                thread_id,
                0 as $ty,
                0 as $acc,
                |acc, x, y| acc.wrapping_add((*x as $acc).wrapping_mul(*y as $acc)),
                |acc| acc as $ty,
            )
        }
    };
}

matmul_signed!(matmul_i8, i8, i64);
matmul_signed!(matmul_i16, i16, i64);
matmul_signed!(matmul_i32, i32, i64);
matmul_signed!(matmul_i64, i64, i128);
matmul_unsigned!(matmul_u8, u8, u64);
matmul_unsigned!(matmul_u16, u16, u64);
matmul_unsigned!(matmul_u32, u32, u64);
matmul_unsigned!(matmul_u64, u64, u128);
