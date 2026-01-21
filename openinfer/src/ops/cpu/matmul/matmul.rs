use anyhow::{anyhow, Result};

use crate::tensor::{Bitset, F16, Tensor, TensorOptions};
use crate::timer::Timer;

pub(crate) fn matmul_dims(a_shape: &[usize], b_shape: &[usize]) -> Result<(usize, usize, usize)> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(anyhow!(
            "matmul expects 2D inputs, got {:?} and {:?}",
            a_shape,
            b_shape
        ));
    }
    let m = a_shape[0];
    let k = a_shape[1];
    let k2 = b_shape[0];
    let n = b_shape[1];
    if k != k2 {
        return Err(anyhow!(
            "matmul inner dims must match, got {:?} and {:?}",
            a_shape,
            b_shape
        ));
    }
    Ok((m, k, n))
}

pub fn matmul_f32(a: &Tensor<f32>, b: &Tensor<f32>, thread_id: usize) -> Result<Tensor<f32>> {
    let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
    let mut out = vec![0.0f32; m * n];
    Timer::start(thread_id);
    for i in 0..m {
        let a_row = i * k;
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a.data[a_row + kk] * b.data[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    Timer::stop(thread_id);
    Tensor::from_vec_with_opts(
        out,
        TensorOptions {
            shape: Some(vec![m, n]),
            ..TensorOptions::default()
        },
    )
}

pub fn matmul_f64(a: &Tensor<f64>, b: &Tensor<f64>, thread_id: usize) -> Result<Tensor<f64>> {
    let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
    let mut out = vec![0.0f64; m * n];
    Timer::start(thread_id);
    for i in 0..m {
        let a_row = i * k;
        for j in 0..n {
            let mut acc = 0.0f64;
            for kk in 0..k {
                acc += a.data[a_row + kk] * b.data[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    Timer::stop(thread_id);
    Tensor::from_vec_with_opts(
        out,
        TensorOptions {
            shape: Some(vec![m, n]),
            ..TensorOptions::default()
        },
    )
}

pub fn matmul_f16(a: &Tensor<F16>, b: &Tensor<F16>, thread_id: usize) -> Result<Tensor<F16>> {
    let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
    let mut out = vec![F16::from_f32(0.0); m * n];
    Timer::start(thread_id);
    for i in 0..m {
        let a_row = i * k;
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a.data[a_row + kk].to_f32() * b.data[kk * n + j].to_f32();
            }
            out[i * n + j] = F16::from_f32(acc);
        }
    }
    Timer::stop(thread_id);
    Tensor::from_vec_with_opts(
        out,
        TensorOptions {
            shape: Some(vec![m, n]),
            ..TensorOptions::default()
        },
    )
}

pub fn matmul_bool(a: &Tensor<bool>, b: &Tensor<bool>, thread_id: usize) -> Result<Tensor<bool>> {
    let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
    let mut out = vec![false; m * n];
    Timer::start(thread_id);
    for i in 0..m {
        let a_row = i * k;
        for j in 0..n {
            let mut acc = false;
            for kk in 0..k {
                if a.data[a_row + kk] && b.data[kk * n + j] {
                    acc = true;
                    break;
                }
            }
            out[i * n + j] = acc;
        }
    }
    Timer::stop(thread_id);
    Tensor::from_vec_with_opts(
        out,
        TensorOptions {
            shape: Some(vec![m, n]),
            ..TensorOptions::default()
        },
    )
}

pub fn matmul_bitset(
    a: &Tensor<Bitset>,
    b: &Tensor<Bitset>,
    thread_id: usize,
) -> Result<Tensor<Bitset>> {
    let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
    let mut out = vec![Bitset { bits: 0 }; m * n];
    Timer::start(thread_id);
    for i in 0..m {
        let a_row = i * k;
        for j in 0..n {
            let mut acc = 0u64;
            for kk in 0..k {
                acc = acc.wrapping_add(
                    (a.data[a_row + kk].bits as u64)
                        .wrapping_mul(b.data[kk * n + j].bits as u64),
                );
            }
            out[i * n + j] = Bitset { bits: acc as u8 };
        }
    }
    Timer::stop(thread_id);
    Tensor::from_vec_with_opts(
        out,
        TensorOptions {
            shape: Some(vec![m, n]),
            ..TensorOptions::default()
        },
    )
}

macro_rules! matmul_signed {
    ($name:ident, $ty:ty, $acc:ty) => {
        pub fn $name(a: &Tensor<$ty>, b: &Tensor<$ty>, thread_id: usize) -> Result<Tensor<$ty>> {
            let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
            let mut out = vec![0 as $ty; m * n];
            Timer::start(thread_id);
            for i in 0..m {
                let a_row = i * k;
                for j in 0..n {
                    let mut acc: $acc = 0;
                    for kk in 0..k {
                        acc = acc.wrapping_add(
                            (a.data[a_row + kk] as $acc)
                                .wrapping_mul(b.data[kk * n + j] as $acc),
                        );
                    }
                    out[i * n + j] = acc as $ty;
                }
            }
            Timer::stop(thread_id);
            Tensor::from_vec_with_opts(
                out,
                TensorOptions {
                    shape: Some(vec![m, n]),
                    ..TensorOptions::default()
                },
            )
        }
    };
}

macro_rules! matmul_unsigned {
    ($name:ident, $ty:ty, $acc:ty) => {
        pub fn $name(a: &Tensor<$ty>, b: &Tensor<$ty>, thread_id: usize) -> Result<Tensor<$ty>> {
            let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
            let mut out = vec![0 as $ty; m * n];
            Timer::start(thread_id);
            for i in 0..m {
                let a_row = i * k;
                for j in 0..n {
                    let mut acc: $acc = 0;
                    for kk in 0..k {
                        acc = acc.wrapping_add(
                            (a.data[a_row + kk] as $acc)
                                .wrapping_mul(b.data[kk * n + j] as $acc),
                        );
                    }
                    out[i * n + j] = acc as $ty;
                }
            }
            Timer::stop(thread_id);
            Tensor::from_vec_with_opts(
                out,
                TensorOptions {
                    shape: Some(vec![m, n]),
                    ..TensorOptions::default()
                },
            )
        }
    };
}

matmul_signed!(matmul_i8, i8, i64);
matmul_signed!(matmul_i16, i16, i64);
matmul_signed!(matmul_i32, i32, i64);
matmul_signed!(matmul_i64, i64, i128);
matmul_unsigned!(matmul_u8, u8, u64);
matmul_unsigned!(matmul_u16, u16, u64);
matmul_unsigned!(matmul_u32, u32, u64);
matmul_unsigned!(matmul_u64, u64, u128);
