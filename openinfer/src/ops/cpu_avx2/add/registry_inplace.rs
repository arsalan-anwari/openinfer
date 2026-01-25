use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::InplaceKernelFn;
use crate::tensor::{DType, TensorValue};

use super::{
    add_inplace_bool, add_inplace_f32, add_inplace_f64, add_inplace_i16, add_inplace_i32,
    add_inplace_i64, add_inplace_i8, add_inplace_u16, add_inplace_u32, add_inplace_u64,
    add_inplace_u8, add_inplace_i4, add_inplace_i2, add_inplace_u4, add_inplace_u2,
};

pub fn supports_add_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::I8, [DType::I8, DType::I8], OpAttrs::None)
            | (DType::I16, [DType::I16, DType::I16], OpAttrs::None)
            | (DType::I32, [DType::I32, DType::I32], OpAttrs::None)
            | (DType::I64, [DType::I64, DType::I64], OpAttrs::None)
            | (DType::F32, [DType::F32, DType::F32], OpAttrs::None)
            | (DType::F64, [DType::F64, DType::F64], OpAttrs::None)
            | (DType::U8, [DType::U8, DType::U8], OpAttrs::None)
            | (DType::U16, [DType::U16, DType::U16], OpAttrs::None)
            | (DType::U32, [DType::U32, DType::U32], OpAttrs::None)
            | (DType::U64, [DType::U64, DType::U64], OpAttrs::None)
            | (DType::Bool, [DType::Bool, DType::Bool], OpAttrs::None)
            | (DType::I4, [DType::I4, DType::I4], OpAttrs::None)
            | (DType::I2, [DType::I2, DType::I2], OpAttrs::None)
            | (DType::U4, [DType::U4, DType::U4], OpAttrs::None)
            | (DType::U2, [DType::U2, DType::U2], OpAttrs::None)
    )
}

pub fn lookup_kernel_cpu_avx2_add_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_add_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", I8, add_inplace_i8)
        }
        (DType::I16, [DType::I16, DType::I16], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", I16, add_inplace_i16)
        }
        (DType::I32, [DType::I32, DType::I32], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", I32, add_inplace_i32)
        }
        (DType::I64, [DType::I64, DType::I64], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", I64, add_inplace_i64)
        }
        (DType::F32, [DType::F32, DType::F32], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", F32, add_inplace_f32)
        }
        (DType::F64, [DType::F64, DType::F64], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", F64, add_inplace_f64)
        }
        (DType::U8, [DType::U8, DType::U8], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", U8, add_inplace_u8)
        }
        (DType::U16, [DType::U16, DType::U16], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", U16, add_inplace_u16)
        }
        (DType::U32, [DType::U32, DType::U32], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", U32, add_inplace_u32)
        }
        (DType::U64, [DType::U64, DType::U64], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", U64, add_inplace_u64)
        }
        (DType::Bool, [DType::Bool, DType::Bool], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", Bool, add_inplace_bool)
        }
        (DType::I4, [DType::I4, DType::I4], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", I4, add_inplace_i4)
        }
        (DType::I2, [DType::I2, DType::I2], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", I2, add_inplace_i2)
        }
        (DType::U4, [DType::U4, DType::U4], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", U4, add_inplace_u4)
        }
        (DType::U2, [DType::U2, DType::U2], &OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "add", U2, add_inplace_u2)
        }
        _ => None,
    }
}
