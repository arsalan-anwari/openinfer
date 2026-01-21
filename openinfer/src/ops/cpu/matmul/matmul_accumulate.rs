use anyhow::Result;

use crate::tensor::{Tensor, TensorOptions};
use crate::timer::Timer;

use super::matmul::matmul_dims;

macro_rules! matmul_accumulate_signed {
    ($name:ident, $in:ty, $out:ty) => {
        pub fn $name(a: &Tensor<$in>, b: &Tensor<$in>, thread_id: usize) -> Result<Tensor<$out>> {
            let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
            let mut out = vec![0 as $out; m * n];
            Timer::start(thread_id);
            for i in 0..m {
                let a_row = i * k;
                for j in 0..n {
                    let mut acc: $out = 0;
                    for kk in 0..k {
                        acc = acc.wrapping_add(
                            (a.data[a_row + kk] as $out)
                                .wrapping_mul(b.data[kk * n + j] as $out),
                        );
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
    };
}

macro_rules! matmul_accumulate_unsigned {
    ($name:ident, $in:ty, $out:ty) => {
        pub fn $name(a: &Tensor<$in>, b: &Tensor<$in>, thread_id: usize) -> Result<Tensor<$out>> {
            let (m, k, n) = matmul_dims(a.shape(), b.shape())?;
            let mut out = vec![0 as $out; m * n];
            Timer::start(thread_id);
            for i in 0..m {
                let a_row = i * k;
                for j in 0..n {
                    let mut acc: $out = 0;
                    for kk in 0..k {
                        acc = acc.wrapping_add(
                            (a.data[a_row + kk] as $out)
                                .wrapping_mul(b.data[kk * n + j] as $out),
                        );
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
    };
}

matmul_accumulate_signed!(matmul_i8_i16, i8, i16);
matmul_accumulate_signed!(matmul_i8_i32, i8, i32);
matmul_accumulate_signed!(matmul_i8_i64, i8, i64);
matmul_accumulate_signed!(matmul_i16_i32, i16, i32);
matmul_accumulate_signed!(matmul_i16_i64, i16, i64);
matmul_accumulate_signed!(matmul_i32_i64, i32, i64);
matmul_accumulate_unsigned!(matmul_u8_u16, u8, u16);
matmul_accumulate_unsigned!(matmul_u8_u32, u8, u32);
matmul_accumulate_unsigned!(matmul_u8_u64, u8, u64);
matmul_accumulate_unsigned!(matmul_u16_u32, u16, u32);
matmul_accumulate_unsigned!(matmul_u16_u64, u16, u64);
matmul_accumulate_unsigned!(matmul_u32_u64, u32, u64);
