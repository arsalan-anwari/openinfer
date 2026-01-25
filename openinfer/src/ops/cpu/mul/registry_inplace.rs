use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::InplaceKernelFn;
use crate::tensor::{DType, TensorValue};

use super::{
    mul_inplace_bf16,
    mul_inplace_bf16_broadcast,
    mul_inplace_bitset,
    mul_inplace_bitset_broadcast,
    mul_inplace_bool,
    mul_inplace_bool_broadcast,
    mul_inplace_f16,
    mul_inplace_f16_broadcast,
    mul_inplace_f32,
    mul_inplace_f32_broadcast,
    mul_inplace_f64,
    mul_inplace_f64_broadcast,
    mul_inplace_f8,
    mul_inplace_f8_broadcast,
    mul_inplace_i16,
    mul_inplace_i16_broadcast,
    mul_inplace_i1,
    mul_inplace_i1_broadcast,
    mul_inplace_i2,
    mul_inplace_i2_broadcast,
    mul_inplace_i32,
    mul_inplace_i32_broadcast,
    mul_inplace_i4,
    mul_inplace_i4_broadcast,
    mul_inplace_i64,
    mul_inplace_i64_broadcast,
    mul_inplace_i8,
    mul_inplace_i8_broadcast,
    mul_inplace_u16,
    mul_inplace_u16_broadcast,
    mul_inplace_u1,
    mul_inplace_u1_broadcast,
    mul_inplace_u2,
    mul_inplace_u2_broadcast,
    mul_inplace_u32,
    mul_inplace_u32_broadcast,
    mul_inplace_u4,
    mul_inplace_u4_broadcast,
    mul_inplace_u64,
    mul_inplace_u64_broadcast,
    mul_inplace_u8,
    mul_inplace_u8_broadcast,
};

#[allow(dead_code)]
pub fn supports_mul_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::I8, [DType::I8, DType::I8], OpAttrs::None)
            | (DType::I16, [DType::I16, DType::I16], OpAttrs::None)
            | (DType::F32, [DType::F32, DType::F32], OpAttrs::None)
            | (DType::F64, [DType::F64, DType::F64], OpAttrs::None)
            | (DType::F16, [DType::F16, DType::F16], OpAttrs::None)
            | (DType::BF16, [DType::BF16, DType::BF16], OpAttrs::None)
            | (DType::F8E5M2, [DType::F8E5M2, DType::F8E5M2], OpAttrs::None)
            | (DType::U8, [DType::U8, DType::U8], OpAttrs::None)
            | (DType::U16, [DType::U16, DType::U16], OpAttrs::None)
            | (DType::I32, [DType::I32, DType::I32], OpAttrs::None)
            | (DType::I64, [DType::I64, DType::I64], OpAttrs::None)
            | (DType::U32, [DType::U32, DType::U32], OpAttrs::None)
            | (DType::U64, [DType::U64, DType::U64], OpAttrs::None)
            | (DType::Bool, [DType::Bool, DType::Bool], OpAttrs::None)
            | (DType::Bitset, [DType::Bitset, DType::Bitset], OpAttrs::None)
            | (DType::I4, [DType::I4, DType::I4], OpAttrs::None)
            | (DType::I2, [DType::I2, DType::I2], OpAttrs::None)
            | (DType::I1, [DType::I1, DType::I1], OpAttrs::None)
            | (DType::U4, [DType::U4, DType::U4], OpAttrs::None)
            | (DType::U2, [DType::U2, DType::U2], OpAttrs::None)
            | (DType::U1, [DType::U1, DType::U1], OpAttrs::None)
    )
}

pub fn lookup_kernel_cpu_mul_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_mul_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", I8, mul_inplace_i8, mul_inplace_i8_broadcast)
        }
        (DType::I16, [DType::I16, DType::I16], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", I16, mul_inplace_i16, mul_inplace_i16_broadcast)
        }
        (DType::F32, [DType::F32, DType::F32], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", F32, mul_inplace_f32, mul_inplace_f32_broadcast)
        }
        (DType::F64, [DType::F64, DType::F64], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", F64, mul_inplace_f64, mul_inplace_f64_broadcast)
        }
        (DType::F16, [DType::F16, DType::F16], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", F16, mul_inplace_f16, mul_inplace_f16_broadcast)
        }
        (DType::BF16, [DType::BF16, DType::BF16], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", BF16, mul_inplace_bf16, mul_inplace_bf16_broadcast)
        }
        (DType::F8E5M2, [DType::F8E5M2, DType::F8E5M2], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", F8E5M2, mul_inplace_f8, mul_inplace_f8_broadcast)
        }
        (DType::U8, [DType::U8, DType::U8], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", U8, mul_inplace_u8, mul_inplace_u8_broadcast)
        }
        (DType::U16, [DType::U16, DType::U16], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", U16, mul_inplace_u16, mul_inplace_u16_broadcast)
        }
        (DType::I32, [DType::I32, DType::I32], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", I32, mul_inplace_i32, mul_inplace_i32_broadcast)
        }
        (DType::I64, [DType::I64, DType::I64], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", I64, mul_inplace_i64, mul_inplace_i64_broadcast)
        }
        (DType::U32, [DType::U32, DType::U32], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", U32, mul_inplace_u32, mul_inplace_u32_broadcast)
        }
        (DType::U64, [DType::U64, DType::U64], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", U64, mul_inplace_u64, mul_inplace_u64_broadcast)
        }
        (DType::Bool, [DType::Bool, DType::Bool], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", Bool, mul_inplace_bool, mul_inplace_bool_broadcast)
        }
        (DType::Bitset, [DType::Bitset, DType::Bitset], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", Bitset, mul_inplace_bitset, mul_inplace_bitset_broadcast)
        }
        (DType::I4, [DType::I4, DType::I4], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", I4, mul_inplace_i4, mul_inplace_i4_broadcast)
        }
        (DType::I2, [DType::I2, DType::I2], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", I2, mul_inplace_i2, mul_inplace_i2_broadcast)
        }
        (DType::I1, [DType::I1, DType::I1], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", I1, mul_inplace_i1, mul_inplace_i1_broadcast)
        }
        (DType::U4, [DType::U4, DType::U4], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", U4, mul_inplace_u4, mul_inplace_u4_broadcast)
        }
        (DType::U2, [DType::U2, DType::U2], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", U2, mul_inplace_u2, mul_inplace_u2_broadcast)
        }
        (DType::U1, [DType::U1, DType::U1], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "mul", U1, mul_inplace_u1, mul_inplace_u1_broadcast)
        }
        _ => None,
    }
}
