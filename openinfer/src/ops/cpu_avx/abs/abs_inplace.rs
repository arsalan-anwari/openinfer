use anyhow::Result;
use crate::timer::Timer;
pub fn abs_inplace_f32(a: &mut [f32], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = v.abs();
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_f64(a: &mut [f64], thread_id: usize) -> Result<()> {
pub fn abs_inplace_f64(a: &mut [f64], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = v.abs();
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_i8(a: &mut [i8], thread_id: usize) -> Result<()> {
pub fn abs_inplace_i8(a: &mut [i8], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = v.abs();
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_i16(a: &mut [i16], thread_id: usize) -> Result<()> {
pub fn abs_inplace_i16(a: &mut [i16], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = v.abs();
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_i32(a: &mut [i32], thread_id: usize) -> Result<()> {
pub fn abs_inplace_i32(a: &mut [i32], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = v.abs();
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_i64(a: &mut [i64], thread_id: usize) -> Result<()> {
pub fn abs_inplace_i64(a: &mut [i64], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = v.abs();
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_u8(_a: &mut [u8], thread_id: usize) -> Result<()> {
pub fn abs_inplace_u8(_a: &mut [u8], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_u16(_a: &mut [u16], thread_id: usize) -> Result<()> {
pub fn abs_inplace_u16(_a: &mut [u16], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_u32(_a: &mut [u32], thread_id: usize) -> Result<()> {
pub fn abs_inplace_u32(_a: &mut [u32], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_u64(_a: &mut [u64], thread_id: usize) -> Result<()> {
pub fn abs_inplace_u64(_a: &mut [u64], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_bool(_a: &mut [bool], thread_id: usize) -> Result<()> {
pub fn abs_inplace_bool(_a: &mut [bool], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    Timer::stop(thread_id);
    Ok(())
}
