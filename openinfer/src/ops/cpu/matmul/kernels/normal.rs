use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::tensor::{Bitset, BF16, F16, F8, Tensor};

use super::common::MatmulElement;

fn matmul_with_data<T: MatmulElement>(
    a_data: &[T],
    a_shape: &[usize],
    a_strides: &[usize],
    b: &Tensor<T>,
    out: &mut Tensor<T>,
) -> Result<()> {
    let b_shape = b.shape();
    let b_strides = b.strides();
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(anyhow!("matmul expects inputs with rank >= 2"));
    }
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();
    let a_m = a_shape[a_rank - 2];
    let a_k = a_shape[a_rank - 1];
    let b_k = b_shape[b_rank - 2];
    let b_n = b_shape[b_rank - 1];
    if a_k != b_k {
        return Err(anyhow!(
            "matmul inner dim mismatch: {} vs {}",
            a_k,
            b_k
        ));
    }
    let a_batch = &a_shape[..a_rank - 2];
    let b_batch = &b_shape[..b_rank - 2];
    let batch_shape = broadcast_shape(a_batch, b_batch)?;
    let batch_rank = batch_shape.len();
    let mut expected = batch_shape.clone();
    expected.push(a_m);
    expected.push(b_n);
    if out.shape() != expected.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match matmul shape {:?}",
            out.shape(),
            expected
        ));
    }
    let out_strides = out.strides();
    let out_batch_strides = out_strides[..batch_rank].to_vec();
    let a_batch_strides = broadcast_strides(a_batch, &a_strides[..a_rank - 2], batch_rank);
    let b_batch_strides = broadcast_strides(b_batch, &b_strides[..b_rank - 2], batch_rank);
    let a_stride_m = a_strides[a_rank - 2];
    let a_stride_k = a_strides[a_rank - 1];
    let b_stride_k = b_strides[b_rank - 2];
    let b_stride_n = b_strides[b_rank - 1];
    let out_stride_m = out_strides[batch_rank];
    let out_stride_n = out_strides[batch_rank + 1];
    for_each_broadcast_index(
        &batch_shape,
        &out_batch_strides,
        &a_batch_strides,
        &b_batch_strides,
        |out_base, a_base, b_base| {
            for m in 0..a_m {
                let a_m_offset = a_base + m * a_stride_m;
                let out_m_offset = out_base + m * out_stride_m;
                for n in 0..b_n {
                    let mut acc = T::zero();
                    for k in 0..a_k {
                        let a_offset = a_m_offset + k * a_stride_k;
                        let b_offset = b_base + k * b_stride_k + n * b_stride_n;
                        acc = acc.add(a_data[a_offset].mul(b.data[b_offset]));
                    }
                    let out_offset = out_m_offset + n * out_stride_n;
                    out.data[out_offset] = acc;
                }
            }
        },
    );
    Ok(())
}

pub fn matmul_normal<T: MatmulElement>(a: &Tensor<T>, b: &Tensor<T>, out: &mut Tensor<T>) -> Result<()> {
    matmul_with_data(&a.data, a.shape(), a.strides(), b, out)
}

pub fn matmul_inplace<T: MatmulElement>(a: &mut Tensor<T>, b: &Tensor<T>) -> Result<()> {
    let a_data = a.data.clone();
    let shape = a.shape().to_vec();
    let strides = a.strides().to_vec();
    matmul_with_data(&a_data, &shape, &strides, b, a)
}

pub fn matmul_f8_normal(a: &Tensor<F8>, b: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_f8_inplace(a: &mut Tensor<F8>, b: &Tensor<F8>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_bf16_normal(a: &Tensor<BF16>, b: &Tensor<BF16>, out: &mut Tensor<BF16>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_bf16_inplace(a: &mut Tensor<BF16>, b: &Tensor<BF16>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_f16_normal(a: &Tensor<F16>, b: &Tensor<F16>, out: &mut Tensor<F16>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_f16_inplace(a: &mut Tensor<F16>, b: &Tensor<F16>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_f32_normal(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_f32_inplace(a: &mut Tensor<f32>, b: &Tensor<f32>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_f64_normal(a: &Tensor<f64>, b: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_f64_inplace(a: &mut Tensor<f64>, b: &Tensor<f64>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_i8_normal(a: &Tensor<i8>, b: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_i8_inplace(a: &mut Tensor<i8>, b: &Tensor<i8>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_i16_normal(a: &Tensor<i16>, b: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_i16_inplace(a: &mut Tensor<i16>, b: &Tensor<i16>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_i32_normal(a: &Tensor<i32>, b: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_i32_inplace(a: &mut Tensor<i32>, b: &Tensor<i32>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_i64_normal(a: &Tensor<i64>, b: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_i64_inplace(a: &mut Tensor<i64>, b: &Tensor<i64>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_u8_normal(a: &Tensor<u8>, b: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_u8_inplace(a: &mut Tensor<u8>, b: &Tensor<u8>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_u16_normal(a: &Tensor<u16>, b: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_u16_inplace(a: &mut Tensor<u16>, b: &Tensor<u16>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_u32_normal(a: &Tensor<u32>, b: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_u32_inplace(a: &mut Tensor<u32>, b: &Tensor<u32>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_u64_normal(a: &Tensor<u64>, b: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_u64_inplace(a: &mut Tensor<u64>, b: &Tensor<u64>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_bool_normal(a: &Tensor<bool>, b: &Tensor<bool>, out: &mut Tensor<bool>) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_bool_inplace(a: &mut Tensor<bool>, b: &Tensor<bool>) -> Result<()> {
    matmul_inplace(a, b)
}

pub fn matmul_bitset_normal(
    a: &Tensor<Bitset>,
    b: &Tensor<Bitset>,
    out: &mut Tensor<Bitset>,
) -> Result<()> {
    matmul_normal(a, b, out)
}

pub fn matmul_bitset_inplace(a: &mut Tensor<Bitset>, b: &Tensor<Bitset>) -> Result<()> {
    matmul_inplace(a, b)
}
