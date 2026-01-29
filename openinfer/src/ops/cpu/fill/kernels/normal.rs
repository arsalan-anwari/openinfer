use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::tensor::{Bitset, BF16, F16, F8, Tensor};

use super::common::{fill_value_bool, fill_value_f64, fill_value_i64, fill_value_u64};

fn ensure_shape<T>(a: &Tensor<T>, out: &Tensor<T>) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    Ok(())
}

pub fn fill_f8_normal(attrs: &OpAttrs, a: &Tensor<F8>, out: &mut Tensor<F8>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_f64(attrs)? as f32;
    let value = F8::from_f32(value);
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_f8_inplace(attrs: &OpAttrs, a: &mut Tensor<F8>) -> Result<()> {
    let value = fill_value_f64(attrs)? as f32;
    let value = F8::from_f32(value);
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_bf16_normal(attrs: &OpAttrs, a: &Tensor<BF16>, out: &mut Tensor<BF16>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_f64(attrs)? as f32;
    let value = BF16::from_f32(value);
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_bf16_inplace(attrs: &OpAttrs, a: &mut Tensor<BF16>) -> Result<()> {
    let value = fill_value_f64(attrs)? as f32;
    let value = BF16::from_f32(value);
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_f16_normal(attrs: &OpAttrs, a: &Tensor<F16>, out: &mut Tensor<F16>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_f64(attrs)? as f32;
    let value = F16::from_f32(value);
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_f16_inplace(attrs: &OpAttrs, a: &mut Tensor<F16>) -> Result<()> {
    let value = fill_value_f64(attrs)? as f32;
    let value = F16::from_f32(value);
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_f32_normal(attrs: &OpAttrs, a: &Tensor<f32>, out: &mut Tensor<f32>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_f64(attrs)? as f32;
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_f32_inplace(attrs: &OpAttrs, a: &mut Tensor<f32>) -> Result<()> {
    let value = fill_value_f64(attrs)? as f32;
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_f64_normal(attrs: &OpAttrs, a: &Tensor<f64>, out: &mut Tensor<f64>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_f64(attrs)?;
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_f64_inplace(attrs: &OpAttrs, a: &mut Tensor<f64>) -> Result<()> {
    let value = fill_value_f64(attrs)?;
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_i8_normal(attrs: &OpAttrs, a: &Tensor<i8>, out: &mut Tensor<i8>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_i64(attrs)? as i8;
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_i8_inplace(attrs: &OpAttrs, a: &mut Tensor<i8>) -> Result<()> {
    let value = fill_value_i64(attrs)? as i8;
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_i16_normal(attrs: &OpAttrs, a: &Tensor<i16>, out: &mut Tensor<i16>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_i64(attrs)? as i16;
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_i16_inplace(attrs: &OpAttrs, a: &mut Tensor<i16>) -> Result<()> {
    let value = fill_value_i64(attrs)? as i16;
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_i32_normal(attrs: &OpAttrs, a: &Tensor<i32>, out: &mut Tensor<i32>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_i64(attrs)? as i32;
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_i32_inplace(attrs: &OpAttrs, a: &mut Tensor<i32>) -> Result<()> {
    let value = fill_value_i64(attrs)? as i32;
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_i64_normal(attrs: &OpAttrs, a: &Tensor<i64>, out: &mut Tensor<i64>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_i64(attrs)?;
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_i64_inplace(attrs: &OpAttrs, a: &mut Tensor<i64>) -> Result<()> {
    let value = fill_value_i64(attrs)?;
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_u8_normal(attrs: &OpAttrs, a: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_u64(attrs)? as u8;
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_u8_inplace(attrs: &OpAttrs, a: &mut Tensor<u8>) -> Result<()> {
    let value = fill_value_u64(attrs)? as u8;
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_u16_normal(attrs: &OpAttrs, a: &Tensor<u16>, out: &mut Tensor<u16>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_u64(attrs)? as u16;
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_u16_inplace(attrs: &OpAttrs, a: &mut Tensor<u16>) -> Result<()> {
    let value = fill_value_u64(attrs)? as u16;
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_u32_normal(attrs: &OpAttrs, a: &Tensor<u32>, out: &mut Tensor<u32>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_u64(attrs)? as u32;
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_u32_inplace(attrs: &OpAttrs, a: &mut Tensor<u32>) -> Result<()> {
    let value = fill_value_u64(attrs)? as u32;
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_u64_normal(attrs: &OpAttrs, a: &Tensor<u64>, out: &mut Tensor<u64>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_u64(attrs)?;
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_u64_inplace(attrs: &OpAttrs, a: &mut Tensor<u64>) -> Result<()> {
    let value = fill_value_u64(attrs)?;
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_bool_normal(attrs: &OpAttrs, a: &Tensor<bool>, out: &mut Tensor<bool>) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_bool(attrs)?;
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_bool_inplace(attrs: &OpAttrs, a: &mut Tensor<bool>) -> Result<()> {
    let value = fill_value_bool(attrs)?;
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_bitset_normal(
    attrs: &OpAttrs,
    a: &Tensor<Bitset>,
    out: &mut Tensor<Bitset>,
) -> Result<()> {
    ensure_shape(a, out)?;
    let value = fill_value_u64(attrs)? as u8;
    let value = Bitset { bits: value };
    for out_slot in &mut out.data {
        *out_slot = value;
    }
    Ok(())
}

pub fn fill_bitset_inplace(attrs: &OpAttrs, a: &mut Tensor<Bitset>) -> Result<()> {
    let value = fill_value_u64(attrs)? as u8;
    let value = Bitset { bits: value };
    for out_slot in &mut a.data {
        *out_slot = value;
    }
    Ok(())
}
