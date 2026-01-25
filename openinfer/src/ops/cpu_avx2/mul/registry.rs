use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, I2, I4, U2, U4, TensorElement, TensorValue};

use super::{
    mul_bool, mul_f32, mul_f64, mul_i16, mul_i32, mul_i64, mul_i8, mul_u16, mul_u32, mul_u64,
    mul_u8, mul_i4_packed, mul_i2_packed, mul_u4_packed, mul_u2_packed,
};

pub fn lookup_kernel_cpu_avx2_mul(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", i8, I8, mul_i8)
        }
        (DType::I16, [DType::I16, DType::I16], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", i16, I16, mul_i16)
        }
        (DType::F32, [DType::F32, DType::F32], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", f32, F32, mul_f32)
        }
        (DType::F64, [DType::F64, DType::F64], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", f64, F64, mul_f64)
        }
        (DType::U8, [DType::U8, DType::U8], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", u8, U8, mul_u8)
        }
        (DType::U16, [DType::U16, DType::U16], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", u16, U16, mul_u16)
        }
        (DType::I32, [DType::I32, DType::I32], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", i32, I32, mul_i32)
        }
        (DType::I64, [DType::I64, DType::I64], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", i64, I64, mul_i64)
        }
        (DType::U32, [DType::U32, DType::U32], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", u32, U32, mul_u32)
        }
        (DType::U64, [DType::U64, DType::U64], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", u64, U64, mul_u64)
        }
        (DType::Bool, [DType::Bool, DType::Bool], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", bool, Bool, mul_bool)
        }
        (DType::I4, [DType::I4, DType::I4], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", I4, I4, mul_i4_packed)
        }
        (DType::I2, [DType::I2, DType::I2], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", I2, I2, mul_i2_packed)
        }
        (DType::U4, [DType::U4, DType::U4], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", U4, U4, mul_u4_packed)
        }
        (DType::U2, [DType::U2, DType::U2], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "mul", U2, U2, mul_u2_packed)
        }
        _ => None,
    }
}
