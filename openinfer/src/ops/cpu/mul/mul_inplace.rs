use anyhow::{anyhow, Result};

use crate::ops::cpu::packed::{packed_binary_signed, packed_binary_unsigned};
use crate::tensor::{BF16, Bitset, F16, F8E5M2, I1, I2, I4, U1, U2, U4};
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

pub fn mul_inplace_bf16(a: &mut [BF16], b: &[BF16], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] = BF16::from_f32(a[i].to_f32() * b[i].to_f32());
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_f8(a: &mut [F8E5M2], b: &[F8E5M2], thread_id: usize) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    for i in 0..a.len() {
        a[i] = F8E5M2::from_f32(a[i].to_f32() * b[i].to_f32());
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

pub fn mul_inplace_i4(
    a: &mut [I4],
    b: &[I4],
    logical_len: usize,
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    let out = packed_binary_signed(4, a, b, logical_len, I4 { bits: 0 }, |x, y| x * y);
    a.copy_from_slice(&out);
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_i2(
    a: &mut [I2],
    b: &[I2],
    logical_len: usize,
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    let out = packed_binary_signed(2, a, b, logical_len, I2 { bits: 0 }, |x, y| x * y);
    a.copy_from_slice(&out);
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_i1(
    a: &mut [I1],
    b: &[I1],
    logical_len: usize,
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    let out = packed_binary_signed(1, a, b, logical_len, I1 { bits: 0 }, |x, y| x * y);
    a.copy_from_slice(&out);
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_u4(
    a: &mut [U4],
    b: &[U4],
    logical_len: usize,
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    let out = packed_binary_unsigned(4, a, b, logical_len, U4 { bits: 0 }, |x, y| x * y);
    a.copy_from_slice(&out);
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_u2(
    a: &mut [U2],
    b: &[U2],
    logical_len: usize,
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    let out = packed_binary_unsigned(2, a, b, logical_len, U2 { bits: 0 }, |x, y| x * y);
    a.copy_from_slice(&out);
    Timer::stop(thread_id);
    Ok(())
}

pub fn mul_inplace_u1(
    a: &mut [U1],
    b: &[U1],
    logical_len: usize,
    thread_id: usize,
) -> Result<()> {
    if a.len() != b.len() {
        return Err(anyhow!("mul inplace shape mismatch"));
    }
    Timer::start(thread_id);
    let out = packed_binary_unsigned(1, a, b, logical_len, U1 { bits: 0 }, |x, y| x * y);
    a.copy_from_slice(&out);
    Timer::stop(thread_id);
    Ok(())
}
