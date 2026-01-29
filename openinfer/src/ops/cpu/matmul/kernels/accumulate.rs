use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_shape, broadcast_strides, for_each_broadcast_index};
use crate::ops::cpu::packed_cpu::{get_bits, sign_extend, PackedBits};
use crate::tensor::{I1, I2, I4, Tensor, U1, U2, U4};

use super::common::{SignedAcc, SignedInput, UnsignedAcc, UnsignedInput};

fn matmul_accumulate_signed<In, Acc>(a: &Tensor<In>, b: &Tensor<In>, out: &mut Tensor<Acc>) -> Result<()>
where
    In: SignedInput,
    Acc: SignedAcc,
{
    let b_shape = b.shape();
    let b_strides = b.strides();
    if a.shape().len() < 2 || b_shape.len() < 2 {
        return Err(anyhow!("matmul expects inputs with rank >= 2"));
    }
    let a_shape = a.shape();
    let a_strides = a.strides();
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
                    let mut acc: i64 = 0;
                    for k in 0..a_k {
                        let a_offset = a_m_offset + k * a_stride_k;
                        let b_offset = b_base + k * b_stride_k + n * b_stride_n;
                        let lhs = a.data[a_offset].to_i64();
                        let rhs = b.data[b_offset].to_i64();
                        acc = acc.wrapping_add(lhs.wrapping_mul(rhs));
                    }
                    let out_offset = out_m_offset + n * out_stride_n;
                    out.data[out_offset] = Acc::from_i64(acc);
                }
            }
        },
    );
    Ok(())
}

fn matmul_accumulate_unsigned<In, Acc>(a: &Tensor<In>, b: &Tensor<In>, out: &mut Tensor<Acc>) -> Result<()>
where
    In: UnsignedInput,
    Acc: UnsignedAcc,
{
    let b_shape = b.shape();
    let b_strides = b.strides();
    if a.shape().len() < 2 || b_shape.len() < 2 {
        return Err(anyhow!("matmul expects inputs with rank >= 2"));
    }
    let a_shape = a.shape();
    let a_strides = a.strides();
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
                    let mut acc: u64 = 0;
                    for k in 0..a_k {
                        let a_offset = a_m_offset + k * a_stride_k;
                        let b_offset = b_base + k * b_stride_k + n * b_stride_n;
                        let lhs = a.data[a_offset].to_u64();
                        let rhs = b.data[b_offset].to_u64();
                        acc = acc.wrapping_add(lhs.wrapping_mul(rhs));
                    }
                    let out_offset = out_m_offset + n * out_stride_n;
                    out.data[out_offset] = Acc::from_u64(acc);
                }
            }
        },
    );
    Ok(())
}

fn matmul_accumulate_packed_signed<T, Acc>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<Acc>,
    width: u8,
) -> Result<()>
where
    T: PackedBits,
    Acc: SignedAcc,
{
    let b_shape = b.shape();
    let b_strides = b.strides();
    if a.shape().len() < 2 || b_shape.len() < 2 {
        return Err(anyhow!("matmul expects inputs with rank >= 2"));
    }
    let a_shape = a.shape();
    let a_strides = a.strides();
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
                    let mut acc: i64 = 0;
                    for k in 0..a_k {
                        let a_offset = a_m_offset + k * a_stride_k;
                        let b_offset = b_base + k * b_stride_k + n * b_stride_n;
                        let lhs = sign_extend(get_bits(&a.data, a_offset, width), width) as i64;
                        let rhs = sign_extend(get_bits(&b.data, b_offset, width), width) as i64;
                        acc = acc.wrapping_add(lhs.wrapping_mul(rhs));
                    }
                    let out_offset = out_m_offset + n * out_stride_n;
                    out.data[out_offset] = Acc::from_i64(acc);
                }
            }
        },
    );
    Ok(())
}

fn matmul_accumulate_packed_unsigned<T, Acc>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<Acc>,
    width: u8,
) -> Result<()>
where
    T: PackedBits,
    Acc: UnsignedAcc,
{
    let b_shape = b.shape();
    let b_strides = b.strides();
    if a.shape().len() < 2 || b_shape.len() < 2 {
        return Err(anyhow!("matmul expects inputs with rank >= 2"));
    }
    let a_shape = a.shape();
    let a_strides = a.strides();
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
                    let mut acc: u64 = 0;
                    for k in 0..a_k {
                        let a_offset = a_m_offset + k * a_stride_k;
                        let b_offset = b_base + k * b_stride_k + n * b_stride_n;
                        let lhs = get_bits(&a.data, a_offset, width) as u64;
                        let rhs = get_bits(&b.data, b_offset, width) as u64;
                        acc = acc.wrapping_add(lhs.wrapping_mul(rhs));
                    }
                    let out_offset = out_m_offset + n * out_stride_n;
                    out.data[out_offset] = Acc::from_u64(acc);
                }
            }
        },
    );
    Ok(())
}

macro_rules! signed_acc_fn {
    ($name:ident, $in:ty, $acc:ty) => {
        pub fn $name(a: &Tensor<$in>, b: &Tensor<$in>, out: &mut Tensor<$acc>) -> Result<()> {
            matmul_accumulate_signed::<$in, $acc>(a, b, out)
        }
    };
}

macro_rules! unsigned_acc_fn {
    ($name:ident, $in:ty, $acc:ty) => {
        pub fn $name(a: &Tensor<$in>, b: &Tensor<$in>, out: &mut Tensor<$acc>) -> Result<()> {
            matmul_accumulate_unsigned::<$in, $acc>(a, b, out)
        }
    };
}

signed_acc_fn!(matmul_i8_accumulate_i16, i8, i16);
signed_acc_fn!(matmul_i8_accumulate_i32, i8, i32);
signed_acc_fn!(matmul_i8_accumulate_i64, i8, i64);
signed_acc_fn!(matmul_i16_accumulate_i32, i16, i32);
signed_acc_fn!(matmul_i16_accumulate_i64, i16, i64);
signed_acc_fn!(matmul_i32_accumulate_i64, i32, i64);

unsigned_acc_fn!(matmul_u8_accumulate_u16, u8, u16);
unsigned_acc_fn!(matmul_u8_accumulate_u32, u8, u32);
unsigned_acc_fn!(matmul_u8_accumulate_u64, u8, u64);
unsigned_acc_fn!(matmul_u16_accumulate_u32, u16, u32);
unsigned_acc_fn!(matmul_u16_accumulate_u64, u16, u64);
unsigned_acc_fn!(matmul_u32_accumulate_u64, u32, u64);

pub fn matmul_i1_accumulate_i8(a: &Tensor<I1>, b: &Tensor<I1>, out: &mut Tensor<i8>) -> Result<()> {
    matmul_accumulate_packed_signed(a, b, out, 1)
}

pub fn matmul_i2_accumulate_i8(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<i8>) -> Result<()> {
    matmul_accumulate_packed_signed(a, b, out, 2)
}

pub fn matmul_i4_accumulate_i8(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<i8>) -> Result<()> {
    matmul_accumulate_packed_signed(a, b, out, 4)
}

pub fn matmul_i1_accumulate_i16(a: &Tensor<I1>, b: &Tensor<I1>, out: &mut Tensor<i16>) -> Result<()> {
    matmul_accumulate_packed_signed(a, b, out, 1)
}

pub fn matmul_i2_accumulate_i16(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<i16>) -> Result<()> {
    matmul_accumulate_packed_signed(a, b, out, 2)
}

pub fn matmul_i4_accumulate_i16(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<i16>) -> Result<()> {
    matmul_accumulate_packed_signed(a, b, out, 4)
}

pub fn matmul_i1_accumulate_i32(a: &Tensor<I1>, b: &Tensor<I1>, out: &mut Tensor<i32>) -> Result<()> {
    matmul_accumulate_packed_signed(a, b, out, 1)
}

pub fn matmul_i2_accumulate_i32(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<i32>) -> Result<()> {
    matmul_accumulate_packed_signed(a, b, out, 2)
}

pub fn matmul_i4_accumulate_i32(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<i32>) -> Result<()> {
    matmul_accumulate_packed_signed(a, b, out, 4)
}

pub fn matmul_i1_accumulate_i64(a: &Tensor<I1>, b: &Tensor<I1>, out: &mut Tensor<i64>) -> Result<()> {
    matmul_accumulate_packed_signed(a, b, out, 1)
}

pub fn matmul_i2_accumulate_i64(a: &Tensor<I2>, b: &Tensor<I2>, out: &mut Tensor<i64>) -> Result<()> {
    matmul_accumulate_packed_signed(a, b, out, 2)
}

pub fn matmul_i4_accumulate_i64(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<i64>) -> Result<()> {
    matmul_accumulate_packed_signed(a, b, out, 4)
}

pub fn matmul_u1_accumulate_u8(a: &Tensor<U1>, b: &Tensor<U1>, out: &mut Tensor<u8>) -> Result<()> {
    matmul_accumulate_packed_unsigned(a, b, out, 1)
}

pub fn matmul_u2_accumulate_u8(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<u8>) -> Result<()> {
    matmul_accumulate_packed_unsigned(a, b, out, 2)
}

pub fn matmul_u4_accumulate_u8(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<u8>) -> Result<()> {
    matmul_accumulate_packed_unsigned(a, b, out, 4)
}

pub fn matmul_u1_accumulate_u16(a: &Tensor<U1>, b: &Tensor<U1>, out: &mut Tensor<u16>) -> Result<()> {
    matmul_accumulate_packed_unsigned(a, b, out, 1)
}

pub fn matmul_u2_accumulate_u16(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<u16>) -> Result<()> {
    matmul_accumulate_packed_unsigned(a, b, out, 2)
}

pub fn matmul_u4_accumulate_u16(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<u16>) -> Result<()> {
    matmul_accumulate_packed_unsigned(a, b, out, 4)
}

pub fn matmul_u1_accumulate_u32(a: &Tensor<U1>, b: &Tensor<U1>, out: &mut Tensor<u32>) -> Result<()> {
    matmul_accumulate_packed_unsigned(a, b, out, 1)
}

pub fn matmul_u2_accumulate_u32(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<u32>) -> Result<()> {
    matmul_accumulate_packed_unsigned(a, b, out, 2)
}

pub fn matmul_u4_accumulate_u32(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<u32>) -> Result<()> {
    matmul_accumulate_packed_unsigned(a, b, out, 4)
}

pub fn matmul_u1_accumulate_u64(a: &Tensor<U1>, b: &Tensor<U1>, out: &mut Tensor<u64>) -> Result<()> {
    matmul_accumulate_packed_unsigned(a, b, out, 1)
}

pub fn matmul_u2_accumulate_u64(a: &Tensor<U2>, b: &Tensor<U2>, out: &mut Tensor<u64>) -> Result<()> {
    matmul_accumulate_packed_unsigned(a, b, out, 2)
}

pub fn matmul_u4_accumulate_u64(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<u64>) -> Result<()> {
    matmul_accumulate_packed_unsigned(a, b, out, 4)
}
