use anyhow::{anyhow, Result};

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

pub fn mul_f32(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!("mul op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    for i in 0..a.len() {
        out.push(a[i] * b[i]);
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
