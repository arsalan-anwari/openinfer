use anyhow::Result;

use crate::tensor::Tensor;
use crate::timer::Timer;

use super::matmul::matmul_dims;

macro_rules! matmul_accumulate_signed {
    ($name:ident, $in:ty, $out:ty) => {
        pub fn $name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            output: Option<&mut [$out]>,
            thread_id: usize,
        ) -> Result<Option<Vec<$out>>> {
            let (batch, m, k, n) = matmul_dims(a.shape(), b.shape())?;
            let len = batch * m * n;
            Timer::start(thread_id);
            let reuse = output.is_some();
            let mut out_storage = if reuse {
                None
            } else {
                Some(vec![0 as $out; len])
            };
            let out = match output {
                Some(out) => {
                    if out.len() != len {
                        return Err(anyhow::anyhow!("matmul output shape mismatch"));
                    }
                    out
                }
                None => out_storage.as_mut().unwrap().as_mut_slice(),
            };
            for batch_idx in 0..batch {
                let a_base = batch_idx * m * k;
                let b_base = batch_idx * k * n;
                let out_base = batch_idx * m * n;
                for i in 0..m {
                    let a_row = a_base + i * k;
                    for j in 0..n {
                        let mut acc: $out = 0;
                        for kk in 0..k {
                            acc = acc.wrapping_add(
                                (a.data[a_row + kk] as $out)
                                    .wrapping_mul(b.data[b_base + kk * n + j] as $out),
                            );
                        }
                        out[out_base + i * n + j] = acc;
                    }
                }
            }
            Timer::stop(thread_id);
            if reuse {
                Ok(None)
            } else {
                Ok(out_storage)
            }
        }
    };
}

macro_rules! matmul_accumulate_unsigned {
    ($name:ident, $in:ty, $out:ty) => {
        pub fn $name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            output: Option<&mut [$out]>,
            thread_id: usize,
        ) -> Result<Option<Vec<$out>>> {
            let (batch, m, k, n) = matmul_dims(a.shape(), b.shape())?;
            let len = batch * m * n;
            Timer::start(thread_id);
            let reuse = output.is_some();
            let mut out_storage = if reuse {
                None
            } else {
                Some(vec![0 as $out; len])
            };
            let out = match output {
                Some(out) => {
                    if out.len() != len {
                        return Err(anyhow::anyhow!("matmul output shape mismatch"));
                    }
                    out
                }
                None => out_storage.as_mut().unwrap().as_mut_slice(),
            };
            for batch_idx in 0..batch {
                let a_base = batch_idx * m * k;
                let b_base = batch_idx * k * n;
                let out_base = batch_idx * m * n;
                for i in 0..m {
                    let a_row = a_base + i * k;
                    for j in 0..n {
                        let mut acc: $out = 0;
                        for kk in 0..k {
                            acc = acc.wrapping_add(
                                (a.data[a_row + kk] as $out)
                                    .wrapping_mul(b.data[b_base + kk * n + j] as $out),
                            );
                        }
                        out[out_base + i * n + j] = acc;
                    }
                }
            }
            Timer::stop(thread_id);
            if reuse {
                Ok(None)
            } else {
                Ok(out_storage)
            }
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
