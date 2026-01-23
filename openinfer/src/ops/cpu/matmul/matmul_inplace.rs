use anyhow::{anyhow, Result};

use crate::tensor::{numel, Bitset, F16, Tensor};
use crate::timer::Timer;

use super::matmul::matmul_dims;

fn ensure_output_shape<T>(out: &Tensor<T>, batch_shape: &[usize], m: usize, n: usize) -> Result<()> {
    let mut expected = batch_shape.to_vec();
    expected.push(m);
    expected.push(n);
    if out.shape() != expected.as_slice() {
        return Err(anyhow!(
            "matmul inplace output shape mismatch: expected {:?}, got {:?}",
            expected,
            out.shape()
        ));
    }
    Ok(())
}

fn matmul_broadcast_inplace<T, Acc, F, G>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    thread_id: usize,
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
    ensure_output_shape(a, &batch_shape, m, n)?;
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
    let a_data = a.data.clone();
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
                    let a_val = &a_data[a_row_base + kk * a_stride_k];
                    let b_val = &b.data[b_batch_offset + kk * b_stride_k + j * b_stride_n];
                    acc = accumulate(acc, a_val, b_val);
                }
                a.data[out_base + i * n + j] = finalize(acc);
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

pub fn matmul_inplace_f32(a: &mut Tensor<f32>, b: &Tensor<f32>, thread_id: usize) -> Result<()> {
    matmul_broadcast_inplace(
        a,
        b,
        thread_id,
        0.0f32,
        |acc, x, y| acc + (*x * *y),
        |acc| acc,
    )
}

pub fn matmul_inplace_f64(a: &mut Tensor<f64>, b: &Tensor<f64>, thread_id: usize) -> Result<()> {
    matmul_broadcast_inplace(
        a,
        b,
        thread_id,
        0.0f64,
        |acc, x, y| acc + (*x * *y),
        |acc| acc,
    )
}

pub fn matmul_inplace_f16(a: &mut Tensor<F16>, b: &Tensor<F16>, thread_id: usize) -> Result<()> {
    matmul_broadcast_inplace(
        a,
        b,
        thread_id,
        0.0f32,
        |acc, x, y| acc + x.to_f32() * y.to_f32(),
        |acc| F16::from_f32(acc),
    )
}

pub fn matmul_inplace_bool(a: &mut Tensor<bool>, b: &Tensor<bool>, thread_id: usize) -> Result<()> {
    matmul_broadcast_inplace(
        a,
        b,
        thread_id,
        false,
        |acc, x, y| acc || (*x && *y),
        |acc| acc,
    )
}

pub fn matmul_inplace_bitset(
    a: &mut Tensor<Bitset>,
    b: &Tensor<Bitset>,
    thread_id: usize,
) -> Result<()> {
    matmul_broadcast_inplace(
        a,
        b,
        thread_id,
        0u64,
        |acc, x, y| acc.wrapping_add((x.bits as u64).wrapping_mul(y.bits as u64)),
        |acc| Bitset { bits: acc as u8 },
    )
}

macro_rules! matmul_signed_inplace {
    ($name:ident, $ty:ty, $acc:ty) => {
        pub fn $name(a: &mut Tensor<$ty>, b: &Tensor<$ty>, thread_id: usize) -> Result<()> {
            matmul_broadcast_inplace(
                a,
                b,
                thread_id,
                0 as $acc,
                |acc, x, y| acc.wrapping_add((*x as $acc).wrapping_mul(*y as $acc)),
                |acc| acc as $ty,
            )
        }
    };
}

macro_rules! matmul_unsigned_inplace {
    ($name:ident, $ty:ty, $acc:ty) => {
        pub fn $name(a: &mut Tensor<$ty>, b: &Tensor<$ty>, thread_id: usize) -> Result<()> {
            matmul_broadcast_inplace(
                a,
                b,
                thread_id,
                0 as $acc,
                |acc, x, y| acc.wrapping_add((*x as $acc).wrapping_mul(*y as $acc)),
                |acc| acc as $ty,
            )
        }
    };
}

matmul_signed_inplace!(matmul_inplace_i8, i8, i64);
matmul_signed_inplace!(matmul_inplace_i16, i16, i64);
matmul_signed_inplace!(matmul_inplace_i32, i32, i64);
matmul_signed_inplace!(matmul_inplace_i64, i64, i128);
matmul_unsigned_inplace!(matmul_inplace_u8, u8, u64);
matmul_unsigned_inplace!(matmul_inplace_u16, u16, u64);
matmul_unsigned_inplace!(matmul_inplace_u32, u32, u64);
matmul_unsigned_inplace!(matmul_inplace_u64, u64, u128);
