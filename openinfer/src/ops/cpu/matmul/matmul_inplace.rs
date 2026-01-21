use anyhow::{anyhow, Result};

use crate::tensor::{Bitset, F16, Tensor};
use crate::timer::Timer;

use super::matmul::matmul_dims;

fn ensure_output_len<T>(out: &Tensor<T>, m: usize, n: usize) -> Result<()> {
    let expected = m.saturating_mul(n);
    if out.data.len() != expected {
        return Err(anyhow!(
            "matmul inplace output shape mismatch: expected {} values, got {}",
            expected,
            out.data.len()
        ));
    }
    Ok(())
}

pub fn matmul_inplace_f32(a: &mut Tensor<f32>, b: &Tensor<f32>, thread_id: usize) -> Result<()> {
    let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
    ensure_output_len(a, m, n)?;
    let a_data = a.data.clone();
    Timer::start(thread_id);
    for i in 0..m {
        let a_row = i * k;
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_data[a_row + kk] * b.data[kk * n + j];
            }
            a.data[i * n + j] = acc;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn matmul_inplace_f64(a: &mut Tensor<f64>, b: &Tensor<f64>, thread_id: usize) -> Result<()> {
    let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
    ensure_output_len(a, m, n)?;
    let a_data = a.data.clone();
    Timer::start(thread_id);
    for i in 0..m {
        let a_row = i * k;
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a_data[a_row + kk] * b.data[kk * n + j];
            }
            a.data[i * n + j] = acc;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn matmul_inplace_f16(a: &mut Tensor<F16>, b: &Tensor<F16>, thread_id: usize) -> Result<()> {
    let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
    ensure_output_len(a, m, n)?;
    let a_data = a.data.clone();
    Timer::start(thread_id);
    for i in 0..m {
        let a_row = i * k;
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a_data[a_row + kk].to_f32() * b.data[kk * n + j].to_f32();
            }
            a.data[i * n + j] = F16::from_f32(acc);
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn matmul_inplace_bool(a: &mut Tensor<bool>, b: &Tensor<bool>, thread_id: usize) -> Result<()> {
    let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
    ensure_output_len(a, m, n)?;
    let a_data = a.data.clone();
    Timer::start(thread_id);
    for i in 0..m {
        let a_row = i * k;
        for j in 0..n {
            let mut acc = false;
            for kk in 0..k {
                if a_data[a_row + kk] && b.data[kk * n + j] {
                    acc = true;
                    break;
                }
            }
            a.data[i * n + j] = acc;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn matmul_inplace_bitset(
    a: &mut Tensor<Bitset>,
    b: &Tensor<Bitset>,
    thread_id: usize,
) -> Result<()> {
    let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
    ensure_output_len(a, m, n)?;
    let a_data = a.data.clone();
    Timer::start(thread_id);
    for i in 0..m {
        let a_row = i * k;
        for j in 0..n {
            let mut acc = 0u64;
            for kk in 0..k {
                acc = acc.wrapping_add(
                    (a_data[a_row + kk].bits as u64)
                        .wrapping_mul(b.data[kk * n + j].bits as u64),
                );
            }
            a.data[i * n + j] = Bitset { bits: acc as u8 };
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

macro_rules! matmul_signed_inplace {
    ($name:ident, $ty:ty, $acc:ty) => {
        pub fn $name(a: &mut Tensor<$ty>, b: &Tensor<$ty>, thread_id: usize) -> Result<()> {
            let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
            ensure_output_len(a, m, n)?;
            let a_data = a.data.clone();
            Timer::start(thread_id);
            for i in 0..m {
                let a_row = i * k;
                for j in 0..n {
                    let mut acc: $acc = 0;
                    for kk in 0..k {
                        acc = acc.wrapping_add(
                            (a_data[a_row + kk] as $acc)
                                .wrapping_mul(b.data[kk * n + j] as $acc),
                        );
                    }
                    a.data[i * n + j] = acc as $ty;
                }
            }
            Timer::stop(thread_id);
            Ok(())
        }
    };
}

macro_rules! matmul_unsigned_inplace {
    ($name:ident, $ty:ty, $acc:ty) => {
        pub fn $name(a: &mut Tensor<$ty>, b: &Tensor<$ty>, thread_id: usize) -> Result<()> {
            let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
            ensure_output_len(a, m, n)?;
            let a_data = a.data.clone();
            Timer::start(thread_id);
            for i in 0..m {
                let a_row = i * k;
                for j in 0..n {
                    let mut acc: $acc = 0;
                    for kk in 0..k {
                        acc = acc.wrapping_add(
                            (a_data[a_row + kk] as $acc)
                                .wrapping_mul(b.data[kk * n + j] as $acc),
                        );
                    }
                    a.data[i * n + j] = acc as $ty;
                }
            }
            Timer::stop(thread_id);
            Ok(())
        }
    };
}

matmul_signed_inplace!(matmul_inplace_i8, i8, i64);
matmul_signed_inplace!(matmul_inplace_i16, i16, i64);
matmul_signed_inplace!(matmul_inplace_i32, i32, i64);
matmul_signed_inplace!(matmul_inplace_i64, i64, i128);
matmul_unsigned_inplace!(matmul_inplace_u8, u8, u64);
matmul_unsigned_inplace!(matmul_inplace_u16, u16, u64);
matmul_unsigned_inplace!(matmul_inplace_u32, u32, u64);
matmul_unsigned_inplace!(matmul_inplace_u64, u64, u128);
