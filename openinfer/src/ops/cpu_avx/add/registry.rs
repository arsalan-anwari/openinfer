use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, I2, I4, U2, U4, TensorElement, TensorValue};

use super::{
    add_bool, add_f32, add_f64, add_i16, add_i32, add_i64, add_i8, add_u16, add_u32, add_u64,
    add_u8, add_i4_packed, add_i2_packed, add_u4_packed, add_u2_packed,
};

pub fn lookup_kernel_cpu_avx_add(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", i8, I8, add_i8)
        }
        (DType::I16, [DType::I16, DType::I16], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", i16, I16, add_i16)
        }
        (DType::I32, [DType::I32, DType::I32], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", i32, I32, add_i32)
        }
        (DType::I64, [DType::I64, DType::I64], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", i64, I64, add_i64)
        }
        (DType::F32, [DType::F32, DType::F32], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", f32, F32, add_f32)
        }
        (DType::F64, [DType::F64, DType::F64], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", f64, F64, add_f64)
        }
        (DType::U8, [DType::U8, DType::U8], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", u8, U8, add_u8)
        }
        (DType::U16, [DType::U16, DType::U16], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", u16, U16, add_u16)
        }
        (DType::U32, [DType::U32, DType::U32], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", u32, U32, add_u32)
        }
        (DType::U64, [DType::U64, DType::U64], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", u64, U64, add_u64)
        }
        (DType::Bool, [DType::Bool, DType::Bool], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", bool, Bool, add_bool)
        }
        (DType::I4, [DType::I4, DType::I4], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", I4, I4, add_i4_packed)
        }
        (DType::I2, [DType::I2, DType::I2], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", I2, I2, add_i2_packed)
        }
        (DType::U4, [DType::U4, DType::U4], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", U4, U4, add_u4_packed)
        }
        (DType::U2, [DType::U2, DType::U2], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "add", U2, U2, add_u2_packed)
        }
        _ => None,
    }
}
