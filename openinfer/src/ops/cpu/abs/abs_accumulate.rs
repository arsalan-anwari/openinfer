use anyhow::Result;

use crate::ops::cpu::packed::packed_unary_accumulate_signed;
use crate::tensor::{I1, I2, I4, Tensor, TensorOptions};
use crate::timer::Timer;

macro_rules! abs_accumulate_signed {
    ($name:ident, $in:ty, $out:ty) => {
        pub fn $name(a: &[$in], thread_id: usize) -> Result<Vec<$out>> {
            let mut out = Vec::with_capacity(a.len());
            Timer::start(thread_id);
            for value in a {
                let v = *value as i64;
                let y = if v < 0 { -v } else { v };
                out.push(y as $out);
            }
            Timer::stop(thread_id);
            Ok(out)
        }
    };
}

abs_accumulate_signed!(abs_i8_i16, i8, i16);
abs_accumulate_signed!(abs_i8_i32, i8, i32);
abs_accumulate_signed!(abs_i8_i64, i8, i64);
abs_accumulate_signed!(abs_i16_i32, i16, i32);
abs_accumulate_signed!(abs_i16_i64, i16, i64);
abs_accumulate_signed!(abs_i32_i64, i32, i64);

macro_rules! abs_packed_signed_accumulate {
    ($name:ident, $in:ty, $bits:expr, $out:ty) => {
        pub fn $name(a: &Tensor<$in>, thread_id: usize) -> Result<Tensor<$out>> {
            Timer::start(thread_id);
            let out = packed_unary_accumulate_signed($bits, &a.data, a.numel(), |x| {
                let v = if x < 0 { -x } else { x };
                v as $out
            });
            Timer::stop(thread_id);
            Tensor::from_vec_with_opts(
                out,
                TensorOptions {
                    shape: Some(a.shape().to_vec()),
                    ..TensorOptions::default()
                },
            )
        }
    };
}

abs_packed_signed_accumulate!(abs_i4_i8_packed, I4, 4, i8);
abs_packed_signed_accumulate!(abs_i4_i16_packed, I4, 4, i16);
abs_packed_signed_accumulate!(abs_i4_i32_packed, I4, 4, i32);
abs_packed_signed_accumulate!(abs_i4_i64_packed, I4, 4, i64);
abs_packed_signed_accumulate!(abs_i2_i8_packed, I2, 2, i8);
abs_packed_signed_accumulate!(abs_i2_i16_packed, I2, 2, i16);
abs_packed_signed_accumulate!(abs_i2_i32_packed, I2, 2, i32);
abs_packed_signed_accumulate!(abs_i2_i64_packed, I2, 2, i64);
abs_packed_signed_accumulate!(abs_i1_i8_packed, I1, 1, i8);
abs_packed_signed_accumulate!(abs_i1_i16_packed, I1, 1, i16);
abs_packed_signed_accumulate!(abs_i1_i32_packed, I1, 1, i32);
abs_packed_signed_accumulate!(abs_i1_i64_packed, I1, 1, i64);
