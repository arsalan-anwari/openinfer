use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};
use crate::tensor::{Bitset, F16};
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
