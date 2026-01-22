use anyhow::Result;

use crate::ops::cpu::packed::packed_unary_accumulate_signed;
use crate::tensor::{I1, I2, I4, Tensor};
use crate::timer::Timer;

macro_rules! abs_accumulate_signed {
    ($name:ident, $in:ty, $out:ty) => {
        pub fn $name(
            a: &[$in],
            output: Option<&mut [$out]>,
            thread_id: usize,
        ) -> Result<Option<Vec<$out>>> {
            Timer::start(thread_id);
            if let Some(out) = output {
                if out.len() != a.len() {
                    return Err(anyhow::anyhow!("abs op output shape mismatch"));
                }
                for (idx, value) in a.iter().enumerate() {
                    let v = *value as i64;
                    let y = if v < 0 { -v } else { v };
                    out[idx] = y as $out;
                }
                Timer::stop(thread_id);
                return Ok(None);
            }
            let mut out = vec![0 as $out; a.len()];
            for (idx, value) in a.iter().enumerate() {
                let v = *value as i64;
                let y = if v < 0 { -v } else { v };
                out[idx] = y as $out;
            }
            Timer::stop(thread_id);
            Ok(Some(out))
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
        pub fn $name(
            a: &Tensor<$in>,
            output: Option<&mut [$out]>,
            thread_id: usize,
        ) -> Result<Option<Vec<$out>>> {
            let logical_len = a.numel();
            Timer::start(thread_id);
            if let Some(out) = output {
                if out.len() != logical_len {
                    return Err(anyhow::anyhow!("abs op output shape mismatch"));
                }
                for idx in 0..logical_len {
                    let raw = crate::ops::cpu::packed::packed_read(&a.data, idx, $bits);
                    let x = crate::ops::cpu::packed::sign_extend(raw, $bits);
                    let v = if x < 0 { -x } else { x };
                    out[idx] = v as $out;
                }
                Timer::stop(thread_id);
                return Ok(None);
            }
            let out = packed_unary_accumulate_signed($bits, &a.data, logical_len, |x| {
                let v = if x < 0 { -x } else { x };
                v as $out
            });
            Timer::stop(thread_id);
            Ok(Some(out))
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
