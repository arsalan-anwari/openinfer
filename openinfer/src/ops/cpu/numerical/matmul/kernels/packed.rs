use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::ops::cpu::packed_cpu::{get_bits, set_bits, sign_extend, PackedBits};
use crate::tensor::{I1, I2, I4, Tensor, U1, U2, U4};

fn matmul_packed_with_data<T: PackedBits>(
    a_data: &[T],
    a_shape: &[usize],
    a_strides: &[usize],
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
    signed: bool,
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
                    if signed {
                        let mut acc: i8 = 0;
                        for k in 0..a_k {
                            let a_offset = a_m_offset + k * a_stride_k;
                            let b_offset = b_base + k * b_stride_k + n * b_stride_n;
                            let lhs = sign_extend(get_bits(a_data, a_offset, width), width);
                            let rhs = sign_extend(get_bits(&b.data, b_offset, width), width);
                            acc = acc.wrapping_add(lhs.wrapping_mul(rhs));
                        }
                        let out_offset = out_m_offset + n * out_stride_n;
                        set_bits(&mut out.data, out_offset, width, acc as u8);
                    } else {
                        let mut acc: u8 = 0;
                        for k in 0..a_k {
                            let a_offset = a_m_offset + k * a_stride_k;
                            let b_offset = b_base + k * b_stride_k + n * b_stride_n;
                            let lhs = get_bits(a_data, a_offset, width);
                            let rhs = get_bits(&b.data, b_offset, width);
                            acc = acc.wrapping_add(lhs.wrapping_mul(rhs));
                        }
                        let out_offset = out_m_offset + n * out_stride_n;
                        set_bits(&mut out.data, out_offset, width, acc);
                    }
                }
            }
        },
    );
    Ok(())
}

fn matmul_packed<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
    signed: bool,
) -> Result<()> {
    matmul_packed_with_data(&a.data, a.shape(), a.strides(), b, out, width, signed)
}

fn matmul_packed_inplace<T: PackedBits>(
    a: &mut Tensor<T>,
    b: &Tensor<T>,
    width: u8,
    signed: bool,
) -> Result<()> {
    let a_data = a.data.clone();
    let shape = a.shape().to_vec();
    let strides = a.strides().to_vec();
    matmul_packed_with_data(&a_data, &shape, &strides, b, a, width, signed)
}

pub fn matmul_i1_packed(a: &Tensor<I1>, b: &Tensor<I1>, out: &mut Tensor<I1>) -> Result<()> {
    matmul_packed(a, b, out, 1, true)
}

pub fn matmul_i2_packed(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<I2>) -> Result<()> {
    matmul_packed(a, b, out, 2, true)
}

pub fn matmul_i4_packed(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    matmul_packed(a, b, out, 4, true)
}

pub fn matmul_u1_packed(a: &Tensor<U1>, b: &Tensor<U1>, out: &mut Tensor<U1>) -> Result<()> {
    matmul_packed(a, b, out, 1, false)
}

pub fn matmul_u2_packed(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<U2>) -> Result<()> {
    matmul_packed(a, b, out, 2, false)
}

pub fn matmul_u4_packed(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    matmul_packed(a, b, out, 4, false)
}

pub fn matmul_i1_packed_inplace(a: &mut Tensor<I1>, b: &Tensor<I1>) -> Result<()> {
    matmul_packed_inplace(a, b, 1, true)
}

pub fn matmul_i2_packed_inplace(a: &mut Tensor<I2>, b: &Tensor<I2>) -> Result<()> {
    matmul_packed_inplace(a, b, 2, true)
}

pub fn matmul_i4_packed_inplace(a: &mut Tensor<I4>, b: &Tensor<I4>) -> Result<()> {
    matmul_packed_inplace(a, b, 4, true)
}

pub fn matmul_u1_packed_inplace(a: &mut Tensor<U1>, b: &Tensor<U1>) -> Result<()> {
    matmul_packed_inplace(a, b, 1, false)
}

pub fn matmul_u2_packed_inplace(a: &mut Tensor<U2>, b: &Tensor<U2>) -> Result<()> {
    matmul_packed_inplace(a, b, 2, false)
}

pub fn matmul_u4_packed_inplace(a: &mut Tensor<U4>, b: &Tensor<U4>) -> Result<()> {
    matmul_packed_inplace(a, b, 4, false)
}
