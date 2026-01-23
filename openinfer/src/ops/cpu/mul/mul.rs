use anyhow::{anyhow, Result};

use crate::ops::cpu::packed::{packed_binary_signed, packed_binary_unsigned};
use crate::tensor::{
    broadcast_shapes, broadcast_strides, compute_strides, numel, Tensor, BF16, Bitset, F16,
    F8E5M2, I1, I2, I4, U1, U2, U4,
};
use crate::timer::Timer;

fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    strides == compute_strides(shape)
}

fn mul_broadcasted<T, F>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out_shape: &[usize],
    mut f: F,
    thread_id: usize,
) -> Result<Vec<T>>
where
    T: Clone,
    F: FnMut(&T, &T) -> T,
{
    let out_len = numel(out_shape);
    if out_len == 0 {
        return Ok(Vec::new());
    }
    let a_strides = broadcast_strides(a.shape(), a.strides(), out_shape)?;
    let b_strides = broadcast_strides(b.shape(), b.strides(), out_shape)?;
    let mut coords = vec![0usize; out_shape.len()];
    let mut out = Vec::with_capacity(out_len);
    Timer::start(thread_id);
    for _ in 0..out_len {
        let mut a_offset = 0usize;
        let mut b_offset = 0usize;
        for (dim, coord) in coords.iter().enumerate() {
            a_offset = a_offset.saturating_add(coord.saturating_mul(a_strides[dim]));
            b_offset = b_offset.saturating_add(coord.saturating_mul(b_strides[dim]));
        }
        out.push(f(&a.data[a_offset], &b.data[b_offset]));
        for dim in (0..coords.len()).rev() {
            coords[dim] += 1;
            if coords[dim] < out_shape[dim] {
                break;
            }
            coords[dim] = 0;
        }
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_tensor_i8(a: &Tensor<i8>, b: &Tensor<i8>, thread_id: usize) -> Result<(Vec<i8>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_i8(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| x * y, thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_i16(a: &Tensor<i16>, b: &Tensor<i16>, thread_id: usize) -> Result<(Vec<i16>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_i16(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| x * y, thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_f32(a: &Tensor<f32>, b: &Tensor<f32>, thread_id: usize) -> Result<(Vec<f32>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_f32(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| x * y, thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_f64(a: &Tensor<f64>, b: &Tensor<f64>, thread_id: usize) -> Result<(Vec<f64>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_f64(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| x * y, thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_f16(a: &Tensor<F16>, b: &Tensor<F16>, thread_id: usize) -> Result<(Vec<F16>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_f16(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| F16::from_f32(x.to_f32() * y.to_f32()), thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_bf16(a: &Tensor<BF16>, b: &Tensor<BF16>, thread_id: usize) -> Result<(Vec<BF16>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_bf16(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| BF16::from_f32(x.to_f32() * y.to_f32()), thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_f8(a: &Tensor<F8E5M2>, b: &Tensor<F8E5M2>, thread_id: usize) -> Result<(Vec<F8E5M2>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_f8(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| F8E5M2::from_f32(x.to_f32() * y.to_f32()), thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_u8(a: &Tensor<u8>, b: &Tensor<u8>, thread_id: usize) -> Result<(Vec<u8>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_u8(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| x * y, thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_u16(a: &Tensor<u16>, b: &Tensor<u16>, thread_id: usize) -> Result<(Vec<u16>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_u16(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| x * y, thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_i32(a: &Tensor<i32>, b: &Tensor<i32>, thread_id: usize) -> Result<(Vec<i32>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_i32(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| x * y, thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_i64(a: &Tensor<i64>, b: &Tensor<i64>, thread_id: usize) -> Result<(Vec<i64>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_i64(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| x * y, thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_u32(a: &Tensor<u32>, b: &Tensor<u32>, thread_id: usize) -> Result<(Vec<u32>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_u32(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| x * y, thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_u64(a: &Tensor<u64>, b: &Tensor<u64>, thread_id: usize) -> Result<(Vec<u64>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_u64(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| x * y, thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_bool(a: &Tensor<bool>, b: &Tensor<bool>, thread_id: usize) -> Result<(Vec<bool>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_bool(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(a, b, &out_shape, |x, y| *x && *y, thread_id)?;
    Ok((out, out_shape))
}

pub fn mul_tensor_bitset(
    a: &Tensor<Bitset>,
    b: &Tensor<Bitset>,
    thread_id: usize,
) -> Result<(Vec<Bitset>, Vec<usize>)> {
    let out_shape = if a.shape() == b.shape() {
        a.shape().to_vec()
    } else {
        broadcast_shapes(a.shape(), b.shape())?
    };
    if a.shape() == b.shape()
        && is_contiguous(a.shape(), a.strides())
        && is_contiguous(b.shape(), b.strides())
    {
        return Ok((mul_bitset(&a.data, &b.data, thread_id)?, out_shape));
    }
    let out = mul_broadcasted(
        a,
        b,
        &out_shape,
        |x, y| Bitset {
            bits: x.bits.wrapping_mul(y.bits),
        },
        thread_id,
    )?;
    Ok((out, out_shape))
}

pub fn mul_i8(a: &[i8], b: &[i8], thread_id: usize) -> Result<Vec<i8>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i16(a: &[i16], b: &[i16], thread_id: usize) -> Result<Vec<i16>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_f32(a: &[f32], b: &[f32], thread_id: usize) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_f64(a: &[f64], b: &[f64], thread_id: usize) -> Result<Vec<f64>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_f16(a: &[F16], b: &[F16], thread_id: usize) -> Result<Vec<F16>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(F16::from_f32(a[i].to_f32() * b[i].to_f32()));
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_bf16(a: &[BF16], b: &[BF16], thread_id: usize) -> Result<Vec<BF16>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(BF16::from_f32(a[i].to_f32() * b[i].to_f32()));
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_f8(a: &[F8E5M2], b: &[F8E5M2], thread_id: usize) -> Result<Vec<F8E5M2>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(F8E5M2::from_f32(a[i].to_f32() * b[i].to_f32()));
    }
    Timer::stop(thread_id);
    Ok(out)
}
pub fn mul_u8(a: &[u8], b: &[u8], thread_id: usize) -> Result<Vec<u8>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u16(a: &[u16], b: &[u16], thread_id: usize) -> Result<Vec<u16>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i32(a: &[i32], b: &[i32], thread_id: usize) -> Result<Vec<i32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i64(a: &[i64], b: &[i64], thread_id: usize) -> Result<Vec<i64>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u32(a: &[u32], b: &[u32], thread_id: usize) -> Result<Vec<u32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u64(a: &[u64], b: &[u64], thread_id: usize) -> Result<Vec<u64>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_bool(a: &[bool], b: &[bool], thread_id: usize) -> Result<Vec<bool>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] && b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_bitset(a: &[Bitset], b: &[Bitset], thread_id: usize) -> Result<Vec<Bitset>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(Bitset {
            bits: a[i].bits.wrapping_mul(b[i].bits),
        });
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i4(a: &[I4], b: &[I4], logical_len: usize, thread_id: usize) -> Result<Vec<I4>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    Timer::start(thread_id);
    let out = packed_binary_signed(4, a, b, logical_len, I4 { bits: 0 }, |x, y| x * y);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i2(a: &[I2], b: &[I2], logical_len: usize, thread_id: usize) -> Result<Vec<I2>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    Timer::start(thread_id);
    let out = packed_binary_signed(2, a, b, logical_len, I2 { bits: 0 }, |x, y| x * y);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_i1(a: &[I1], b: &[I1], logical_len: usize, thread_id: usize) -> Result<Vec<I1>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    Timer::start(thread_id);
    let out = packed_binary_signed(1, a, b, logical_len, I1 { bits: 0 }, |x, y| x * y);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u4(a: &[U4], b: &[U4], logical_len: usize, thread_id: usize) -> Result<Vec<U4>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    Timer::start(thread_id);
    let out = packed_binary_unsigned(4, a, b, logical_len, U4 { bits: 0 }, |x, y| x * y);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u2(a: &[U2], b: &[U2], logical_len: usize, thread_id: usize) -> Result<Vec<U2>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    Timer::start(thread_id);
    let out = packed_binary_unsigned(2, a, b, logical_len, U2 { bits: 0 }, |x, y| x * y);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn mul_u1(a: &[U1], b: &[U1], logical_len: usize, thread_id: usize) -> Result<Vec<U1>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    Timer::start(thread_id);
    let out = packed_binary_unsigned(1, a, b, logical_len, U1 { bits: 0 }, |x, y| x * y);
    Timer::stop(thread_id);
    Ok(out)
}
