use anyhow::{anyhow, Result};

use crate::timer::Timer;

pub fn add_i8(a: &[i8], b: &[i8], thread_id: usize) -> Result<Vec<i8>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_i16(a: &[i16], b: &[i16], thread_id: usize) -> Result<Vec<i16>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_f32(a: &[f32], b: &[f32], thread_id: usize) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_f64(a: &[f64], b: &[f64], thread_id: usize) -> Result<Vec<f64>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_u8(a: &[u8], b: &[u8], thread_id: usize) -> Result<Vec<u8>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_u16(a: &[u16], b: &[u16], thread_id: usize) -> Result<Vec<u16>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_i32(a: &[i32], b: &[i32], thread_id: usize) -> Result<Vec<i32>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_i64(a: &[i64], b: &[i64], thread_id: usize) -> Result<Vec<i64>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_u32(a: &[u32], b: &[u32], thread_id: usize) -> Result<Vec<u32>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_u64(a: &[u64], b: &[u64], thread_id: usize) -> Result<Vec<u64>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] + b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn add_bool(a: &[bool], b: &[bool], thread_id: usize) -> Result<Vec<bool>> {
    if a.len() != b.len() {
        return Err(anyhow!("add op shape mismatch"));
    }
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for i in 0..a.len() {
        out.push(a[i] || b[i]);
    }
    Timer::stop(thread_id);
    Ok(out)
}
