use anyhow::{anyhow, Result};

use crate::ops::cpu::packed::{packed_binary_accumulate_signed, packed_binary_accumulate_unsigned};
use crate::tensor::{I1, I2, I4, U1, U2, U4, Tensor};
use crate::timer::Timer;

macro_rules! add_accumulate {
    ($name:ident, $in:ty, $out:ty) => {
        pub fn $name(
            a: &[$in],
            b: &[$in],
            output: Option<&mut [$out]>,
            thread_id: usize,
        ) -> Result<Option<Vec<$out>>> {
            if a.len() != b.len() {
                return Err(anyhow!("add op shape mismatch"));
            }
            Timer::start(thread_id);
            if let Some(out) = output {
                if out.len() != a.len() {
                    return Err(anyhow!("add op output shape mismatch"));
                }
                for i in 0..a.len() {
                    out[i] = a[i] as $out + b[i] as $out;
                }
                Timer::stop(thread_id);
                return Ok(None);
            }
            let mut out = vec![0 as $out; a.len()];
            for i in 0..a.len() {
                out[i] = a[i] as $out + b[i] as $out;
            }
            Timer::stop(thread_id);
            Ok(Some(out))
        }
    };
}

add_accumulate!(add_i8_i16, i8, i16);
add_accumulate!(add_i8_i32, i8, i32);
add_accumulate!(add_i8_i64, i8, i64);
add_accumulate!(add_i16_i32, i16, i32);
add_accumulate!(add_i16_i64, i16, i64);
add_accumulate!(add_i32_i64, i32, i64);
add_accumulate!(add_u8_u16, u8, u16);
add_accumulate!(add_u8_u32, u8, u32);
add_accumulate!(add_u8_u64, u8, u64);
add_accumulate!(add_u16_u32, u16, u32);
add_accumulate!(add_u16_u64, u16, u64);
add_accumulate!(add_u32_u64, u32, u64);

macro_rules! add_packed_signed_accumulate {
    ($name:ident, $in:ty, $bits:expr, $out:ty) => {
        pub fn $name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            output: Option<&mut [$out]>,
            thread_id: usize,
        ) -> Result<Option<Vec<$out>>> {
            if a.shape() != b.shape() {
                return Err(anyhow!("add op shape mismatch"));
            }
            Timer::start(thread_id);
            let logical_len = a.numel();
            if let Some(out) = output {
                if out.len() != logical_len {
                    return Err(anyhow!("add op output shape mismatch"));
                }
                for idx in 0..logical_len {
                    let x = crate::ops::cpu::packed::sign_extend(
                        crate::ops::cpu::packed::packed_read(&a.data, idx, $bits),
                        $bits,
                    );
                    let y = crate::ops::cpu::packed::sign_extend(
                        crate::ops::cpu::packed::packed_read(&b.data, idx, $bits),
                        $bits,
                    );
                    out[idx] = (x as i16 + y as i16) as $out;
                }
                Timer::stop(thread_id);
                return Ok(None);
            }
            let out = packed_binary_accumulate_signed($bits, &a.data, &b.data, logical_len, |x, y| {
                (x as i16 + y as i16) as $out
            });
            Timer::stop(thread_id);
            Ok(Some(out))
        }
    };
}

macro_rules! add_packed_unsigned_accumulate {
    ($name:ident, $in:ty, $bits:expr, $out:ty) => {
        pub fn $name(
            a: &Tensor<$in>,
            b: &Tensor<$in>,
            output: Option<&mut [$out]>,
            thread_id: usize,
        ) -> Result<Option<Vec<$out>>> {
            if a.shape() != b.shape() {
                return Err(anyhow!("add op shape mismatch"));
            }
            Timer::start(thread_id);
            let logical_len = a.numel();
            if let Some(out) = output {
                if out.len() != logical_len {
                    return Err(anyhow!("add op output shape mismatch"));
                }
                for idx in 0..logical_len {
                    let x = crate::ops::cpu::packed::packed_read(&a.data, idx, $bits);
                    let y = crate::ops::cpu::packed::packed_read(&b.data, idx, $bits);
                    out[idx] = (x as u16 + y as u16) as $out;
                }
                Timer::stop(thread_id);
                return Ok(None);
            }
            let out = packed_binary_accumulate_unsigned($bits, &a.data, &b.data, logical_len, |x, y| {
                (x as u16 + y as u16) as $out
            });
            Timer::stop(thread_id);
            Ok(Some(out))
        }
    };
}

add_packed_signed_accumulate!(add_i4_i8_packed, I4, 4, i8);
add_packed_signed_accumulate!(add_i4_i16_packed, I4, 4, i16);
add_packed_signed_accumulate!(add_i4_i32_packed, I4, 4, i32);
add_packed_signed_accumulate!(add_i4_i64_packed, I4, 4, i64);
add_packed_signed_accumulate!(add_i2_i8_packed, I2, 2, i8);
add_packed_signed_accumulate!(add_i2_i16_packed, I2, 2, i16);
add_packed_signed_accumulate!(add_i2_i32_packed, I2, 2, i32);
add_packed_signed_accumulate!(add_i2_i64_packed, I2, 2, i64);
add_packed_signed_accumulate!(add_i1_i8_packed, I1, 1, i8);
add_packed_signed_accumulate!(add_i1_i16_packed, I1, 1, i16);
add_packed_signed_accumulate!(add_i1_i32_packed, I1, 1, i32);
add_packed_signed_accumulate!(add_i1_i64_packed, I1, 1, i64);
add_packed_unsigned_accumulate!(add_u4_u8_packed, U4, 4, u8);
add_packed_unsigned_accumulate!(add_u4_u16_packed, U4, 4, u16);
add_packed_unsigned_accumulate!(add_u4_u32_packed, U4, 4, u32);
add_packed_unsigned_accumulate!(add_u4_u64_packed, U4, 4, u64);
add_packed_unsigned_accumulate!(add_u2_u8_packed, U2, 2, u8);
add_packed_unsigned_accumulate!(add_u2_u16_packed, U2, 2, u16);
add_packed_unsigned_accumulate!(add_u2_u32_packed, U2, 2, u32);
add_packed_unsigned_accumulate!(add_u2_u64_packed, U2, 2, u64);
add_packed_unsigned_accumulate!(add_u1_u8_packed, U1, 1, u8);
add_packed_unsigned_accumulate!(add_u1_u16_packed, U1, 1, u16);
add_packed_unsigned_accumulate!(add_u1_u32_packed, U1, 1, u32);
add_packed_unsigned_accumulate!(add_u1_u64_packed, U1, 1, u64);
