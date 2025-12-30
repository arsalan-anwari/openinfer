use anyhow::Result;

use crate::tensor::{Bitset, F16};

pub mod registry;

pub fn abs_i8(a: &[i8]) -> Result<Vec<i8>> {
    let mut out = Vec::with_capacity(a.len());
    for v in a {
        out.push(v.abs());
    }
    Ok(out)
}

pub fn abs_i16(a: &[i16]) -> Result<Vec<i16>> {
    let mut out = Vec::with_capacity(a.len());
    for v in a {
        out.push(v.abs());
    }
    Ok(out)
}

pub fn abs_f32(a: &[f32]) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(a.len());
    for v in a {
        out.push(v.abs());
    }
    Ok(out)
}

pub fn abs_f64(a: &[f64]) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(a.len());
    for v in a {
        out.push(v.abs());
    }
    Ok(out)
}

pub fn abs_u8(a: &[u8]) -> Result<Vec<u8>> {
    Ok(a.to_vec())
}

pub fn abs_u16(a: &[u16]) -> Result<Vec<u16>> {
    Ok(a.to_vec())
}

pub fn abs_i32(a: &[i32]) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(a.len());
    for v in a {
        out.push(v.abs());
    }
    Ok(out)
}

pub fn abs_i64(a: &[i64]) -> Result<Vec<i64>> {
    let mut out = Vec::with_capacity(a.len());
    for v in a {
        out.push(v.abs());
    }
    Ok(out)
}

pub fn abs_u32(a: &[u32]) -> Result<Vec<u32>> {
    Ok(a.to_vec())
}

pub fn abs_u64(a: &[u64]) -> Result<Vec<u64>> {
    Ok(a.to_vec())
}

pub fn abs_bool(a: &[bool]) -> Result<Vec<bool>> {
    Ok(a.to_vec())
}

pub fn abs_bitset(_a: &[Bitset]) -> Result<Vec<Bitset>> {
    println!("cpu abs_bitset: stub");
    Ok(Vec::new())
}

pub fn abs_f16(_a: &[F16]) -> Result<Vec<F16>> {
    println!("cpu abs_f16: stub");
    Ok(Vec::new())
}
