use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::InplaceKernelFn;
use crate::tensor::{DType, TensorValue};

use super::{
    mul_inplace_bool, mul_inplace_f32, mul_inplace_f64, mul_inplace_i16, mul_inplace_i32,
    mul_inplace_i64, mul_inplace_i8, mul_inplace_u16, mul_inplace_u32, mul_inplace_u64,
    mul_inplace_u8, mul_inplace_i4, mul_inplace_i2, mul_inplace_u4, mul_inplace_u2,
};

pub fn supports_mul_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::I8, [DType::I8, DType::I8], OpAttrs::None)
            | (DType::I16, [DType::I16, DType::I16], OpAttrs::None)
            | (DType::F32, [DType::F32, DType::F32], OpAttrs::None)
            | (DType::F64, [DType::F64, DType::F64], OpAttrs::None)
            | (DType::U8, [DType::U8, DType::U8], OpAttrs::None)
            | (DType::U16, [DType::U16, DType::U16], OpAttrs::None)
            | (DType::I32, [DType::I32, DType::I32], OpAttrs::None)
            | (DType::I64, [DType::I64, DType::I64], OpAttrs::None)
            | (DType::U32, [DType::U32, DType::U32], OpAttrs::None)
            | (DType::U64, [DType::U64, DType::U64], OpAttrs::None)
            | (DType::Bool, [DType::Bool, DType::Bool], OpAttrs::None)
            | (DType::I4, [DType::I4, DType::I4], OpAttrs::None)
            | (DType::I2, [DType::I2, DType::I2], OpAttrs::None)
            | (DType::U4, [DType::U4, DType::U4], OpAttrs::None)
            | (DType::U2, [DType::U2, DType::U2], OpAttrs::None)
    )
}

pub fn lookup_kernel_cpu_avx2_mul_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_mul_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", I8, mul_inplace_i8)
        }
        (DType::I16, [DType::I16, DType::I16], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", I16, mul_inplace_i16)
        }
        (DType::F32, [DType::F32, DType::F32], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", F32, mul_inplace_f32)
        }
        (DType::F64, [DType::F64, DType::F64], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", F64, mul_inplace_f64)
        }
        (DType::U8, [DType::U8, DType::U8], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", U8, mul_inplace_u8)
        }
        (DType::U16, [DType::U16, DType::U16], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", U16, mul_inplace_u16)
        }
        (DType::I32, [DType::I32, DType::I32], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", I32, mul_inplace_i32)
        }
        (DType::I64, [DType::I64, DType::I64], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", I64, mul_inplace_i64)
        }
        (DType::U32, [DType::U32, DType::U32], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", U32, mul_inplace_u32)
        }
        (DType::U64, [DType::U64, DType::U64], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", U64, mul_inplace_u64)
        }
        (DType::Bool, [DType::Bool, DType::Bool], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", Bool, mul_inplace_bool)
        }
        (DType::I4, [DType::I4, DType::I4], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", I4, mul_inplace_i4)
        }
        (DType::I2, [DType::I2, DType::I2], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", I2, mul_inplace_i2)
        }
        (DType::U4, [DType::U4, DType::U4], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", U4, mul_inplace_u4)
        }
        (DType::U2, [DType::U2, DType::U2], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "mul", U2, mul_inplace_u2)
        }
        _ => None,
    }
}
