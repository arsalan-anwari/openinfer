use anyhow::Result;

use crate::tensor::{Bitset, F16};
use crate::timer::Timer;

pub mod registry;

pub fn abs_i8(a: &[i8], thread_id: usize) -> Result<Vec<i8>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(v.abs());
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i16(a: &[i16], thread_id: usize) -> Result<Vec<i16>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(v.abs());
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_f32(a: &[f32], thread_id: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(v.abs());
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_f64(a: &[f64], thread_id: usize) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(v.abs());
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_u8(a: &[u8], thread_id: usize) -> Result<Vec<u8>> {
    let out = a.to_vec();
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_u16(a: &[u16], thread_id: usize) -> Result<Vec<u16>> {
    let out = a.to_vec();
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i32(a: &[i32], thread_id: usize) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(v.abs());
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_i64(a: &[i64], thread_id: usize) -> Result<Vec<i64>> {
    let mut out = Vec::with_capacity(a.len());
    Timer::start(thread_id);
    for v in a {
        out.push(v.abs());
    }
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_u32(a: &[u32], thread_id: usize) -> Result<Vec<u32>> {
    let out = a.to_vec();
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_u64(a: &[u64], thread_id: usize) -> Result<Vec<u64>> {
    let out = a.to_vec();
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_bool(a: &[bool], thread_id: usize) -> Result<Vec<bool>> {
    let out = a.to_vec();
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(out)
}

pub fn abs_bitset(_a: &[Bitset], thread_id: usize) -> Result<Vec<Bitset>> {
    println!("cpu abs_bitset: stub");
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(Vec::new())
}

pub fn abs_f16(_a: &[F16], thread_id: usize) -> Result<Vec<F16>> {
    println!("cpu abs_f16: stub");
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(Vec::new())
}
