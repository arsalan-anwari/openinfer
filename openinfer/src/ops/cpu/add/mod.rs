use anyhow::{anyhow, Result};

use crate::tensor::{Bitset, F16};

pub mod registry;

pub fn add_i8(a: &[i8], b: &[i8]) -> Result<Vec<i8>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Ok(out)
}

pub fn add_i16(a: &[i16], b: &[i16]) -> Result<Vec<i16>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Ok(out)
}

pub fn add_f32(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Ok(out)
}

pub fn add_f64(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Ok(out)
}

pub fn add_u8(a: &[u8], b: &[u8]) -> Result<Vec<u8>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Ok(out)
}

pub fn add_u16(a: &[u16], b: &[u16]) -> Result<Vec<u16>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Ok(out)
}

pub fn add_i32(a: &[i32], b: &[i32]) -> Result<Vec<i32>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Ok(out)
}

pub fn add_i64(a: &[i64], b: &[i64]) -> Result<Vec<i64>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Ok(out)
}

pub fn add_u32(a: &[u32], b: &[u32]) -> Result<Vec<u32>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Ok(out)
}

pub fn add_u64(a: &[u64], b: &[u64]) -> Result<Vec<u64>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Ok(out)
}

pub fn add_bool(a: &[bool], b: &[bool]) -> Result<Vec<bool>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] || b[i]);
    }
    Ok(out)
}

pub fn add_bitset(_a: &[Bitset], _b: &[Bitset]) -> Result<Vec<Bitset>> {
    println!("cpu add_bitset: stub");
    Ok(Vec::new())
}

pub fn add_f16(_a: &[F16], _b: &[F16]) -> Result<Vec<F16>> {
    println!("cpu add_f16: stub");
    Ok(Vec::new())
}
