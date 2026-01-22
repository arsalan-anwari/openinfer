use anyhow::{anyhow, Result};

use crate::tensor::{Bitset, F16, Tensor};
use crate::timer::Timer;

use super::matmul::matmul_dims;

fn ensure_output_len<T>(out: &Tensor<T>, batch: usize, m: usize, n: usize) -> Result<()> {
    let expected = batch.saturating_mul(m).saturating_mul(n);
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
    let (batch, m, k, n) = matmul_dims(a.shape(), b.shape())?;
    ensure_output_len(a, batch, m, n)?;
    let a_data = a.data.clone();
    Timer::start(thread_id);
    for batch_idx in 0..batch {
        let a_base = batch_idx * m * k;
        let b_base = batch_idx * k * n;
        let out_base = batch_idx * m * n;
        for i in 0..m {
            let a_row = a_base + i * k;
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_data[a_row + kk] * b.data[b_base + kk * n + j];
                }
                a.data[out_base + i * n + j] = acc;
            }
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn matmul_inplace_f64(a: &mut Tensor<f64>, b: &Tensor<f64>, thread_id: usize) -> Result<()> {
    let (batch, m, k, n) = matmul_dims(a.shape(), b.shape())?;
    ensure_output_len(a, batch, m, n)?;
    let a_data = a.data.clone();
    Timer::start(thread_id);
    for batch_idx in 0..batch {
        let a_base = batch_idx * m * k;
        let b_base = batch_idx * k * n;
        let out_base = batch_idx * m * n;
        for i in 0..m {
            let a_row = a_base + i * k;
            for j in 0..n {
                let mut acc = 0.0f64;
                for kk in 0..k {
                    acc += a_data[a_row + kk] * b.data[b_base + kk * n + j];
                }
                a.data[out_base + i * n + j] = acc;
            }
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn matmul_inplace_f16(a: &mut Tensor<F16>, b: &Tensor<F16>, thread_id: usize) -> Result<()> {
    let (batch, m, k, n) = matmul_dims(a.shape(), b.shape())?;
    ensure_output_len(a, batch, m, n)?;
    let a_data = a.data.clone();
    Timer::start(thread_id);
    for batch_idx in 0..batch {
        let a_base = batch_idx * m * k;
        let b_base = batch_idx * k * n;
        let out_base = batch_idx * m * n;
        for i in 0..m {
            let a_row = a_base + i * k;
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_data[a_row + kk].to_f32() * b.data[b_base + kk * n + j].to_f32();
                }
                a.data[out_base + i * n + j] = F16::from_f32(acc);
            }
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

pub fn matmul_inplace_bool(a: &mut Tensor<bool>, b: &Tensor<bool>, thread_id: usize) -> Result<()> {
    let (batch, m, k, n) = matmul_dims(a.shape(), b.shape())?;
    ensure_output_len(a, batch, m, n)?;
    let a_data = a.data.clone();
    Timer::start(thread_id);
    for batch_idx in 0..batch {
        let a_base = batch_idx * m * k;
        let b_base = batch_idx * k * n;
        let out_base = batch_idx * m * n;
        for i in 0..m {
            let a_row = a_base + i * k;
            for j in 0..n {
                let mut acc = false;
                for kk in 0..k {
                    if a_data[a_row + kk] && b.data[b_base + kk * n + j] {
                        acc = true;
                        break;
                    }
                }
                a.data[out_base + i * n + j] = acc;
            }
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
    let (batch, m, k, n) = matmul_dims(a.shape(), b.shape())?;
    ensure_output_len(a, batch, m, n)?;
    let a_data = a.data.clone();
    Timer::start(thread_id);
    for batch_idx in 0..batch {
        let a_base = batch_idx * m * k;
        let b_base = batch_idx * k * n;
        let out_base = batch_idx * m * n;
        for i in 0..m {
            let a_row = a_base + i * k;
            for j in 0..n {
                let mut acc = 0u64;
                for kk in 0..k {
                    acc = acc.wrapping_add(
                        (a_data[a_row + kk].bits as u64)
                            .wrapping_mul(b.data[b_base + kk * n + j].bits as u64),
                    );
                }
                a.data[out_base + i * n + j] = Bitset { bits: acc as u8 };
            }
        }
    }
    Timer::stop(thread_id);
    Ok(())
}

macro_rules! matmul_signed_inplace {
    ($name:ident, $ty:ty, $acc:ty) => {
        pub fn $name(a: &mut Tensor<$ty>, b: &Tensor<$ty>, thread_id: usize) -> Result<()> {
            let (batch, m, k, n) = matmul_dims(a.shape(), b.shape())?;
            ensure_output_len(a, batch, m, n)?;
            let a_data = a.data.clone();
            Timer::start(thread_id);
            for batch_idx in 0..batch {
                let a_base = batch_idx * m * k;
                let b_base = batch_idx * k * n;
                let out_base = batch_idx * m * n;
                for i in 0..m {
                    let a_row = a_base + i * k;
                    for j in 0..n {
                        let mut acc: $acc = 0;
                        for kk in 0..k {
                            acc = acc.wrapping_add(
                                (a_data[a_row + kk] as $acc)
                                    .wrapping_mul(b.data[b_base + kk * n + j] as $acc),
                            );
                        }
                        a.data[out_base + i * n + j] = acc as $ty;
                    }
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
            let (batch, m, k, n) = matmul_dims(a.shape(), b.shape())?;
            ensure_output_len(a, batch, m, n)?;
            let a_data = a.data.clone();
            Timer::start(thread_id);
            for batch_idx in 0..batch {
                let a_base = batch_idx * m * k;
                let b_base = batch_idx * k * n;
                let out_base = batch_idx * m * n;
                for i in 0..m {
                    let a_row = a_base + i * k;
                    for j in 0..n {
                        let mut acc: $acc = 0;
                        for kk in 0..k {
                            acc = acc.wrapping_add(
                                (a_data[a_row + kk] as $acc)
                                    .wrapping_mul(b.data[b_base + kk * n + j] as $acc),
                            );
                        }
                        a.data[out_base + i * n + j] = acc as $ty;
                    }
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
