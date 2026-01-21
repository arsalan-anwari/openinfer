use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};
use crate::ops::cpu::packed::{packed_fill_signed, packed_fill_unsigned};
use crate::tensor::{BF16, Bitset, F16, F8E5M2, I1, I2, I4, U1, U2, U4};
use crate::timer::Timer;

fn fill_value(attrs: &OpAttrs) -> Result<AttrValue> {
    match attrs {
        OpAttrs::Fill { value } => match value {
            AttrValue::Float(_)
            | AttrValue::Int(_)
            | AttrValue::UInt(_)
            | AttrValue::Bool(_) => Ok(value.clone()),
            AttrValue::Var(_) => Err(anyhow!("fill expects resolved value")),
        },
        _ => Err(anyhow!("fill op expects fill attributes")),
    }
}

pub fn fill_f32(attrs: &OpAttrs, a: &[f32], thread_id: usize) -> Result<Vec<f32>> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => val,
        _ => return Err(anyhow!("fill expects f32 value")),
    };
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for _ in a {
        out.push(value);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_f64(attrs: &OpAttrs, a: &[f64], thread_id: usize) -> Result<Vec<f64>> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => val as f64,
        _ => return Err(anyhow!("fill expects f64 value")),
    };
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for _ in a {
        out.push(value);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_f16(attrs: &OpAttrs, a: &[F16], thread_id: usize) -> Result<Vec<F16>> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => F16::from_f32(val),
        _ => return Err(anyhow!("fill expects f16 value")),
    };
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for _ in a {
        out.push(value);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_bf16(attrs: &OpAttrs, a: &[BF16], thread_id: usize) -> Result<Vec<BF16>> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => BF16::from_f32(val),
        _ => return Err(anyhow!("fill expects bf16 value")),
    };
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for _ in a {
        out.push(value);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_f8(attrs: &OpAttrs, a: &[F8E5M2], thread_id: usize) -> Result<Vec<F8E5M2>> {
    let value = match fill_value(attrs)? {
        AttrValue::Float(val) => F8E5M2::from_f32(val),
        _ => return Err(anyhow!("fill expects f8 value")),
    };
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for _ in a {
        out.push(value);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_i8(attrs: &OpAttrs, a: &[i8], thread_id: usize) -> Result<Vec<i8>> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i8,
        _ => return Err(anyhow!("fill expects i8 value")),
    };
    Timer::start(thread_id);
    let out = vec![value; a.len()];
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_i16(attrs: &OpAttrs, a: &[i16], thread_id: usize) -> Result<Vec<i16>> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i16,
        _ => return Err(anyhow!("fill expects i16 value")),
    };
    Timer::start(thread_id);
    let out = vec![value; a.len()];
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_i32(attrs: &OpAttrs, a: &[i32], thread_id: usize) -> Result<Vec<i32>> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i32,
        _ => return Err(anyhow!("fill expects i32 value")),
    };
    Timer::start(thread_id);
    let out = vec![value; a.len()];
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_i64(attrs: &OpAttrs, a: &[i64], thread_id: usize) -> Result<Vec<i64>> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val,
        _ => return Err(anyhow!("fill expects i64 value")),
    };
    Timer::start(thread_id);
    let out = vec![value; a.len()];
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_u8(attrs: &OpAttrs, a: &[u8], thread_id: usize) -> Result<Vec<u8>> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u8,
        AttrValue::Int(val) => {
            if val < 0 {
                return Err(anyhow!("fill expects u8 value"));
            }
            val as u8
        }
        _ => return Err(anyhow!("fill expects u8 value")),
    };
    Timer::start(thread_id);
    let out = vec![value; a.len()];
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_u16(attrs: &OpAttrs, a: &[u16], thread_id: usize) -> Result<Vec<u16>> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u16,
        AttrValue::Int(val) => {
            if val < 0 {
                return Err(anyhow!("fill expects u16 value"));
            }
            val as u16
        }
        _ => return Err(anyhow!("fill expects u16 value")),
    };
    Timer::start(thread_id);
    let out = vec![value; a.len()];
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_u32(attrs: &OpAttrs, a: &[u32], thread_id: usize) -> Result<Vec<u32>> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u32,
        AttrValue::Int(val) => {
            if val < 0 {
                return Err(anyhow!("fill expects u32 value"));
            }
            val as u32
        }
        _ => return Err(anyhow!("fill expects u32 value")),
    };
    Timer::start(thread_id);
    let out = vec![value; a.len()];
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_u64(attrs: &OpAttrs, a: &[u64], thread_id: usize) -> Result<Vec<u64>> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val,
        AttrValue::Int(val) => {
            if val < 0 {
                return Err(anyhow!("fill expects u64 value"));
            }
            val as u64
        }
        _ => return Err(anyhow!("fill expects u64 value")),
    };
    Timer::start(thread_id);
    let out = vec![value; a.len()];
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_bool(attrs: &OpAttrs, a: &[bool], thread_id: usize) -> Result<Vec<bool>> {
    let value = match fill_value(attrs)? {
        AttrValue::Bool(val) => val,
        _ => return Err(anyhow!("fill expects bool value")),
    };
    Timer::start(thread_id);
    let out = vec![value; a.len()];
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_bitset(
    attrs: &OpAttrs,
    a: &[Bitset],
    thread_id: usize,
) -> Result<Vec<Bitset>> {
    let value = fill_value(attrs)?;
    let bit = match value {
        AttrValue::UInt(val) => {
            if val == 0 {
                0u8
            } else {
                !0u8
            }
        }
        AttrValue::Int(val) => {
            if val == 0 {
                0u8
            } else {
                !0u8
            }
        }
        AttrValue::Bool(val) => {
            if val {
                !0u8
            } else {
                0u8
            }
        }
        _ => return Err(anyhow!("fill expects bitset value")),
    };
    Timer::start(thread_id);
    let out = vec![Bitset { bits: bit }; a.len()];
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_i4(attrs: &OpAttrs, logical_len: usize, thread_id: usize) -> Result<Vec<I4>> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i32,
        _ => return Err(anyhow!("fill expects i4 value")),
    };
    Timer::start(thread_id);
    let out = packed_fill_signed(4, logical_len, value, I4 { bits: 0 });
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_i2(attrs: &OpAttrs, logical_len: usize, thread_id: usize) -> Result<Vec<I2>> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i32,
        _ => return Err(anyhow!("fill expects i2 value")),
    };
    Timer::start(thread_id);
    let out = packed_fill_signed(2, logical_len, value, I2 { bits: 0 });
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_i1(attrs: &OpAttrs, logical_len: usize, thread_id: usize) -> Result<Vec<I1>> {
    let value = match fill_value(attrs)? {
        AttrValue::Int(val) => val as i32,
        _ => return Err(anyhow!("fill expects i1 value")),
    };
    Timer::start(thread_id);
    let out = packed_fill_signed(1, logical_len, value, I1 { bits: 0 });
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_u4(attrs: &OpAttrs, logical_len: usize, thread_id: usize) -> Result<Vec<U4>> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u32,
        AttrValue::Int(val) => {
            if val < 0 {
                return Err(anyhow!("fill expects u4 value"));
            }
            val as u32
        }
        _ => return Err(anyhow!("fill expects u4 value")),
    };
    Timer::start(thread_id);
    let out = packed_fill_unsigned(4, logical_len, value, U4 { bits: 0 });
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_u2(attrs: &OpAttrs, logical_len: usize, thread_id: usize) -> Result<Vec<U2>> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u32,
        AttrValue::Int(val) => {
            if val < 0 {
                return Err(anyhow!("fill expects u2 value"));
            }
            val as u32
        }
        _ => return Err(anyhow!("fill expects u2 value")),
    };
    Timer::start(thread_id);
    let out = packed_fill_unsigned(2, logical_len, value, U2 { bits: 0 });
    Timer::stop(thread_id);
    Ok(out)
}

pub fn fill_u1(attrs: &OpAttrs, logical_len: usize, thread_id: usize) -> Result<Vec<U1>> {
    let value = match fill_value(attrs)? {
        AttrValue::UInt(val) => val as u32,
        AttrValue::Int(val) => {
            if val < 0 {
                return Err(anyhow!("fill expects u1 value"));
            }
            val as u32
        }
        _ => return Err(anyhow!("fill expects u1 value")),
    };
    Timer::start(thread_id);
    let out = packed_fill_unsigned(1, logical_len, value, U1 { bits: 0 });
    Timer::stop(thread_id);
    Ok(out)
}
