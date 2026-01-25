use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};
use crate::ops::cpu::broadcast::{ensure_same_len_unary, ensure_same_shape_unary, is_contiguous};
use crate::ops::cpu::packed::{packed_storage_len, packed_write, PackedByte};
use crate::tensor::{BF16, Bitset, F16, F8E5M2, Tensor, I1, I2, I4, U1, U2, U4};
use crate::timer::Timer;

fn fill_value(attrs: &OpAttrs) -> Result<AttrValue> {
    match attrs {
        OpAttrs::Fill { value } => match value {
            AttrValue::Float(_)
            | AttrValue::Double(_)
            | AttrValue::Int(_)
            | AttrValue::UInt(_)
            | AttrValue::Bool(_) => Ok(value.clone()),
            AttrValue::Var(_) => Err(anyhow!("fill expects resolved value")),
        },
        _ => Err(anyhow!("fill op expects fill attributes")),
    }
}

fn fill_unary<T>(a: &Tensor<T>, out: &mut Tensor<T>, value: T, thread_id: usize) -> Result<()>
where
    T: Clone,
{
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("fill op requires contiguous tensors"));
    }
    ensure_same_len_unary(a, out)?;
    Timer::start(thread_id);
    for slot in out.data.iter_mut() {
        *slot = value.clone();
    }
    Timer::stop(thread_id);
    Ok(())
}

fn fill_inplace_value<T>(out: &mut Tensor<T>, value: T, thread_id: usize) -> Result<()>
where
    T: Clone,
{
    if !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("fill op requires contiguous tensors"));
    }
    Timer::start(thread_id);
    for slot in out.data.iter_mut() {
        *slot = value.clone();
    }
    Timer::stop(thread_id);
    Ok(())
}

fn fill_packed_signed<T: PackedByte + Copy>(
    bits: u8,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    raw: i8,
    thread_id: usize,
) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("fill op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    let storage_len = packed_storage_len(bits, logical_len);
    if a.data.len() != storage_len || out.data.len() != storage_len {
        return Err(anyhow!("fill op packed data length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        packed_write(&mut out.data, idx, bits, raw as u8);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn fill_packed_signed_inplace<T: PackedByte + Copy>(
    bits: u8,
    out: &mut Tensor<T>,
    raw: i8,
    thread_id: usize,
) -> Result<()> {
    if !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("fill op requires contiguous packed tensors"));
    }
    let logical_len = out.numel();
    let storage_len = packed_storage_len(bits, logical_len);
    if out.data.len() != storage_len {
        return Err(anyhow!("fill op packed data length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        packed_write(&mut out.data, idx, bits, raw as u8);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn fill_packed_unsigned<T: PackedByte + Copy>(
    bits: u8,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    raw: u8,
    thread_id: usize,
) -> Result<()> {
    ensure_same_shape_unary(a, out)?;
    if !is_contiguous(a.shape(), a.strides()) || !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("fill op requires contiguous packed tensors"));
    }
    let logical_len = a.numel();
    let storage_len = packed_storage_len(bits, logical_len);
    if a.data.len() != storage_len || out.data.len() != storage_len {
        return Err(anyhow!("fill op packed data length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        packed_write(&mut out.data, idx, bits, raw);
    }
    Timer::stop(thread_id);
    Ok(())
}

fn fill_packed_unsigned_inplace<T: PackedByte + Copy>(
    bits: u8,
    out: &mut Tensor<T>,
    raw: u8,
    thread_id: usize,
) -> Result<()> {
    if !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("fill op requires contiguous packed tensors"));
    }
    let logical_len = out.numel();
    let storage_len = packed_storage_len(bits, logical_len);
    if out.data.len() != storage_len {
        return Err(anyhow!("fill op packed data length mismatch"));
    }
    Timer::start(thread_id);
    for idx in 0..logical_len {
        packed_write(&mut out.data, idx, bits, raw);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn fill_f32(attrs: &OpAttrs, a: &Tensor<f32>, out: &mut Tensor<f32>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => val,
        AttrValue::Double(val) => val as f32,
        _ => return Err(anyhow!("fill expects f32 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_f64(attrs: &OpAttrs, a: &Tensor<f64>, out: &mut Tensor<f64>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => val as f64,
        AttrValue::Double(val) => val,
        _ => return Err(anyhow!("fill expects f64 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_f16(attrs: &OpAttrs, a: &Tensor<F16>, out: &mut Tensor<F16>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => F16::from_f32(val),
        AttrValue::Double(val) => F16::from_f32(val as f32),
        _ => return Err(anyhow!("fill expects f16 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_bf16(attrs: &OpAttrs, a: &Tensor<BF16>, out: &mut Tensor<BF16>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => BF16::from_f32(val),
        AttrValue::Double(val) => BF16::from_f32(val as f32),
        _ => return Err(anyhow!("fill expects bf16 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_f8(attrs: &OpAttrs, a: &Tensor<F8E5M2>, out: &mut Tensor<F8E5M2>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => F8E5M2::from_f32(val),
        AttrValue::Double(val) => F8E5M2::from_f32(val as f32),
        _ => return Err(anyhow!("fill expects f8 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_i8(attrs: &OpAttrs, a: &Tensor<i8>, out: &mut Tensor<i8>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i8,
        _ => return Err(anyhow!("fill expects i8 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_i16(attrs: &OpAttrs, a: &Tensor<i16>, out: &mut Tensor<i16>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i16,
        _ => return Err(anyhow!("fill expects i16 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_i32(attrs: &OpAttrs, a: &Tensor<i32>, out: &mut Tensor<i32>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i32,
        _ => return Err(anyhow!("fill expects i32 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_i64(attrs: &OpAttrs, a: &Tensor<i64>, out: &mut Tensor<i64>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i64,
        _ => return Err(anyhow!("fill expects i64 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_u8(attrs: &OpAttrs, a: &Tensor<u8>, out: &mut Tensor<u8>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u8,
        AttrValue::Int(val) if (0..=u8::MAX as i64).contains(&val) => val as u8,
        _ => return Err(anyhow!("fill expects u8 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_u16(attrs: &OpAttrs, a: &Tensor<u16>, out: &mut Tensor<u16>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u16,
        AttrValue::Int(val) if (0..=u16::MAX as i64).contains(&val) => val as u16,
        _ => return Err(anyhow!("fill expects u16 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_u32(attrs: &OpAttrs, a: &Tensor<u32>, out: &mut Tensor<u32>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u32,
        AttrValue::Int(val) if (0..=u32::MAX as i64).contains(&val) => val as u32,
        _ => return Err(anyhow!("fill expects u32 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_u64(attrs: &OpAttrs, a: &Tensor<u64>, out: &mut Tensor<u64>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u64,
        AttrValue::Int(val) if val >= 0 => val as u64,
        _ => return Err(anyhow!("fill expects u64 value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_bool(attrs: &OpAttrs, a: &Tensor<bool>, out: &mut Tensor<bool>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Bool(val) => val,
        _ => return Err(anyhow!("fill expects bool value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_bitset(attrs: &OpAttrs, a: &Tensor<Bitset>, out: &mut Tensor<Bitset>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => Bitset { bits: val as u8 },
        AttrValue::Int(val) => Bitset { bits: val as u8 },
        _ => return Err(anyhow!("fill expects bitset value")),
    };
    fill_unary(a, out, value, thread_id)
}

pub fn fill_i4(attrs: &OpAttrs, a: &Tensor<I4>, out: &mut Tensor<I4>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i8,
        _ => return Err(anyhow!("fill expects i4 value")),
    };
    fill_packed_signed(4, a, out, value, thread_id)
}

pub fn fill_i2(attrs: &OpAttrs, a: &Tensor<I2>, out: &mut Tensor<I2>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i8,
        _ => return Err(anyhow!("fill expects i2 value")),
    };
    fill_packed_signed(2, a, out, value, thread_id)
}

pub fn fill_i1(attrs: &OpAttrs, a: &Tensor<I1>, out: &mut Tensor<I1>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i8,
        _ => return Err(anyhow!("fill expects i1 value")),
    };
    fill_packed_signed(1, a, out, value, thread_id)
}

pub fn fill_u4(attrs: &OpAttrs, a: &Tensor<U4>, out: &mut Tensor<U4>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u8,
        AttrValue::Int(val) if (0..=u8::MAX as i64).contains(&val) => val as u8,
        _ => return Err(anyhow!("fill expects u4 value")),
    };
    fill_packed_unsigned(4, a, out, value, thread_id)
}

pub fn fill_u2(attrs: &OpAttrs, a: &Tensor<U2>, out: &mut Tensor<U2>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u8,
        AttrValue::Int(val) if (0..=u8::MAX as i64).contains(&val) => val as u8,
        _ => return Err(anyhow!("fill expects u2 value")),
    };
    fill_packed_unsigned(2, a, out, value, thread_id)
}

pub fn fill_u1(attrs: &OpAttrs, a: &Tensor<U1>, out: &mut Tensor<U1>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u8,
        AttrValue::Int(val) if (0..=u8::MAX as i64).contains(&val) => val as u8,
        _ => return Err(anyhow!("fill expects u1 value")),
    };
    fill_packed_unsigned(1, a, out, value, thread_id)
}

pub fn fill_inplace_f32(attrs: &OpAttrs, out: &mut Tensor<f32>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => val,
        AttrValue::Double(val) => val as f32,
        _ => return Err(anyhow!("fill expects f32 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_f64(attrs: &OpAttrs, out: &mut Tensor<f64>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => val as f64,
        AttrValue::Double(val) => val,
        _ => return Err(anyhow!("fill expects f64 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_f16(attrs: &OpAttrs, out: &mut Tensor<F16>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => F16::from_f32(val),
        AttrValue::Double(val) => F16::from_f32(val as f32),
        _ => return Err(anyhow!("fill expects f16 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_bf16(attrs: &OpAttrs, out: &mut Tensor<BF16>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => BF16::from_f32(val),
        AttrValue::Double(val) => BF16::from_f32(val as f32),
        _ => return Err(anyhow!("fill expects bf16 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_f8(attrs: &OpAttrs, out: &mut Tensor<F8E5M2>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => F8E5M2::from_f32(val),
        AttrValue::Double(val) => F8E5M2::from_f32(val as f32),
        _ => return Err(anyhow!("fill expects f8 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_i8(attrs: &OpAttrs, out: &mut Tensor<i8>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i8,
        _ => return Err(anyhow!("fill expects i8 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_i16(attrs: &OpAttrs, out: &mut Tensor<i16>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i16,
        _ => return Err(anyhow!("fill expects i16 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_i32(attrs: &OpAttrs, out: &mut Tensor<i32>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i32,
        _ => return Err(anyhow!("fill expects i32 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_i64(attrs: &OpAttrs, out: &mut Tensor<i64>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i64,
        _ => return Err(anyhow!("fill expects i64 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_u8(attrs: &OpAttrs, out: &mut Tensor<u8>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u8,
        AttrValue::Int(val) if (0..=u8::MAX as i64).contains(&val) => val as u8,
        _ => return Err(anyhow!("fill expects u8 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_u16(attrs: &OpAttrs, out: &mut Tensor<u16>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u16,
        AttrValue::Int(val) if (0..=u16::MAX as i64).contains(&val) => val as u16,
        _ => return Err(anyhow!("fill expects u16 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_u32(attrs: &OpAttrs, out: &mut Tensor<u32>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u32,
        AttrValue::Int(val) if (0..=u32::MAX as i64).contains(&val) => val as u32,
        _ => return Err(anyhow!("fill expects u32 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_u64(attrs: &OpAttrs, out: &mut Tensor<u64>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u64,
        AttrValue::Int(val) if val >= 0 => val as u64,
        _ => return Err(anyhow!("fill expects u64 value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_bool(attrs: &OpAttrs, out: &mut Tensor<bool>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Bool(val) => val,
        _ => return Err(anyhow!("fill expects bool value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_bitset(attrs: &OpAttrs, out: &mut Tensor<Bitset>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => Bitset { bits: val as u8 },
        AttrValue::Int(val) => Bitset { bits: val as u8 },
        _ => return Err(anyhow!("fill expects bitset value")),
    };
    fill_inplace_value(out, value, thread_id)
}

pub fn fill_inplace_i4(attrs: &OpAttrs, out: &mut Tensor<I4>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i8,
        _ => return Err(anyhow!("fill expects i4 value")),
    };
    fill_packed_signed_inplace(4, out, value, thread_id)
}

pub fn fill_inplace_i2(attrs: &OpAttrs, out: &mut Tensor<I2>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i8,
        _ => return Err(anyhow!("fill expects i2 value")),
    };
    fill_packed_signed_inplace(2, out, value, thread_id)
}

pub fn fill_inplace_i1(attrs: &OpAttrs, out: &mut Tensor<I1>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i8,
        _ => return Err(anyhow!("fill expects i1 value")),
    };
    fill_packed_signed_inplace(1, out, value, thread_id)
}

pub fn fill_inplace_u4(attrs: &OpAttrs, out: &mut Tensor<U4>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u8,
        AttrValue::Int(val) if (0..=u8::MAX as i64).contains(&val) => val as u8,
        _ => return Err(anyhow!("fill expects u4 value")),
    };
    fill_packed_unsigned_inplace(4, out, value, thread_id)
}

pub fn fill_inplace_u2(attrs: &OpAttrs, out: &mut Tensor<U2>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u8,
        AttrValue::Int(val) if (0..=u8::MAX as i64).contains(&val) => val as u8,
        _ => return Err(anyhow!("fill expects u2 value")),
    };
    fill_packed_unsigned_inplace(2, out, value, thread_id)
}

pub fn fill_inplace_u1(attrs: &OpAttrs, out: &mut Tensor<U1>, thread_id: usize) -> Result<()> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u8,
        AttrValue::Int(val) if (0..=u8::MAX as i64).contains(&val) => val as u8,
        _ => return Err(anyhow!("fill expects u1 value")),
    };
    fill_packed_unsigned_inplace(1, out, value, thread_id)
}
