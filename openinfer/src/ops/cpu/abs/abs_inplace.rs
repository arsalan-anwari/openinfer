use anyhow::Result;

use crate::ops::cpu::packed::packed_unary_signed_inplace;
use crate::tensor::{BF16, F16, F8E5M2, I1, I2, I4};
use crate::timer::Timer;

pub fn abs_inplace_i8(a: &mut [i8], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = v.abs();
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_i16(a: &mut [i16], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = v.abs();
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_f32(a: &mut [f32], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = v.abs();
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_f64(a: &mut [f64], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = v.abs();
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_f16(a: &mut [F16], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = F16::from_f32(v.to_f32().abs());
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_bf16(a: &mut [BF16], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = BF16::from_f32(v.to_f32().abs());
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_f8(a: &mut [F8E5M2], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = F8E5M2::from_f32(v.to_f32().abs());
    }
    Timer::stop(thread_id);
    Ok(())
}
pub fn abs_inplace_i32(a: &mut [i32], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = v.abs();
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_i64(a: &mut [i64], thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    for v in a {
        *v = v.abs();
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_i4(a: &mut [I4], logical_len: usize, thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    packed_unary_signed_inplace(4, a, logical_len, |x| if x < 0 { -x } else { x });
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_i2(a: &mut [I2], logical_len: usize, thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    packed_unary_signed_inplace(2, a, logical_len, |x| if x < 0 { -x } else { x });
    Timer::stop(thread_id);
    Ok(())
}

pub fn abs_inplace_i1(a: &mut [I1], logical_len: usize, thread_id: usize) -> Result<()> {
    Timer::start(thread_id);
    packed_unary_signed_inplace(1, a, logical_len, |x| if x < 0 { -x } else { x });
    Timer::stop(thread_id);
    Ok(())
}

