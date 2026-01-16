use anyhow::{anyhow, Result};

use crate::tensor::{Bitset, F16};
use crate::timer::Timer;

pub fn mul_inplace_i8(a: &mut [i8], b: &[i8], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] *= b[i];
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_i16(a: &mut [i16], b: &[i16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] *= b[i];
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_f32(a: &mut [f32], b: &[f32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] *= b[i];
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_f64(a: &mut [f64], b: &[f64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] *= b[i];
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_f16(a: &mut [F16], b: &[F16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] = F16::from_f32(a[i].to_f32() * b[i].to_f32());
    }
    Timer::stop(thread_id);
    Ok(())
}
pub fn mul_inplace_u8(a: &mut [u8], b: &[u8], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] = a[i].wrapping_mul(b[i]);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_u16(a: &mut [u16], b: &[u16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] = a[i].wrapping_mul(b[i]);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_i32(a: &mut [i32], b: &[i32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] *= b[i];
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_i64(a: &mut [i64], b: &[i64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] *= b[i];
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_u32(a: &mut [u32], b: &[u32], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] = a[i].wrapping_mul(b[i]);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_u64(a: &mut [u64], b: &[u64], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] = a[i].wrapping_mul(b[i]);
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_bool(a: &mut [bool], b: &[bool], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] = a[i] && b[i];
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_bitset(a: &mut [Bitset], b: &[Bitset], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i].bits = a[i].bits.wrapping_mul(b[i].bits);
    }
    Timer::stop(thread_id);
    Ok(())
}
