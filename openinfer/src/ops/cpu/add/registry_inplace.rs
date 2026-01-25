use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::InplaceKernelFn;
use crate::tensor::{DType, TensorValue};

use super::{
    add_inplace_bf16,
    add_inplace_bf16_broadcast,
    add_inplace_bitset,
    add_inplace_bitset_broadcast,
    add_inplace_bool,
    add_inplace_bool_broadcast,
    add_inplace_f16,
    add_inplace_f16_broadcast,
    add_inplace_f32,
    add_inplace_f32_broadcast,
    add_inplace_f64,
    add_inplace_f64_broadcast,
    add_inplace_f8,
    add_inplace_f8_broadcast,
    add_inplace_i16,
    add_inplace_i16_broadcast,
    add_inplace_i1,
    add_inplace_i1_broadcast,
    add_inplace_i2,
    add_inplace_i2_broadcast,
    add_inplace_i32,
    add_inplace_i32_broadcast,
    add_inplace_i4,
    add_inplace_i4_broadcast,
    add_inplace_i64,
    add_inplace_i64_broadcast,
    add_inplace_i8,
    add_inplace_i8_broadcast,
    add_inplace_u16,
    add_inplace_u16_broadcast,
    add_inplace_u1,
    add_inplace_u1_broadcast,
    add_inplace_u2,
    add_inplace_u2_broadcast,
    add_inplace_u32,
    add_inplace_u32_broadcast,
    add_inplace_u4,
    add_inplace_u4_broadcast,
    add_inplace_u64,
    add_inplace_u64_broadcast,
    add_inplace_u8,
    add_inplace_u8_broadcast,
};

#[allow(dead_code)]
pub fn supports_add_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
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

pub fn lookup_kernel_cpu_add_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_add_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", I8, add_inplace_i8, add_inplace_i8_broadcast)
        }
        (DType::I16, [DType::I16, DType::I16], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", I16, add_inplace_i16, add_inplace_i16_broadcast)
        }
        (DType::F32, [DType::F32, DType::F32], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", F32, add_inplace_f32, add_inplace_f32_broadcast)
        }
        (DType::F64, [DType::F64, DType::F64], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", F64, add_inplace_f64, add_inplace_f64_broadcast)
        }
        (DType::F16, [DType::F16, DType::F16], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", F16, add_inplace_f16, add_inplace_f16_broadcast)
        }
        (DType::BF16, [DType::BF16, DType::BF16], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", BF16, add_inplace_bf16, add_inplace_bf16_broadcast)
        }
        (DType::F8E5M2, [DType::F8E5M2, DType::F8E5M2], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", F8E5M2, add_inplace_f8, add_inplace_f8_broadcast)
        }
        (DType::U8, [DType::U8, DType::U8], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", U8, add_inplace_u8, add_inplace_u8_broadcast)
        }
        (DType::U16, [DType::U16, DType::U16], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", U16, add_inplace_u16, add_inplace_u16_broadcast)
        }
        (DType::I32, [DType::I32, DType::I32], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", I32, add_inplace_i32, add_inplace_i32_broadcast)
        }
        (DType::I64, [DType::I64, DType::I64], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", I64, add_inplace_i64, add_inplace_i64_broadcast)
        }
        (DType::U32, [DType::U32, DType::U32], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", U32, add_inplace_u32, add_inplace_u32_broadcast)
        }
        (DType::U64, [DType::U64, DType::U64], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", U64, add_inplace_u64, add_inplace_u64_broadcast)
        }
        (DType::Bool, [DType::Bool, DType::Bool], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", Bool, add_inplace_bool, add_inplace_bool_broadcast)
        }
        (DType::Bitset, [DType::Bitset, DType::Bitset], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", Bitset, add_inplace_bitset, add_inplace_bitset_broadcast)
        }
        (DType::I4, [DType::I4, DType::I4], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", I4, add_inplace_i4, add_inplace_i4_broadcast)
        }
        (DType::I2, [DType::I2, DType::I2], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", I2, add_inplace_i2, add_inplace_i2_broadcast)
        }
        (DType::I1, [DType::I1, DType::I1], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", I1, add_inplace_i1, add_inplace_i1_broadcast)
        }
        (DType::U4, [DType::U4, DType::U4], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", U4, add_inplace_u4, add_inplace_u4_broadcast)
        }
        (DType::U2, [DType::U2, DType::U2], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", U2, add_inplace_u2, add_inplace_u2_broadcast)
        }
        (DType::U1, [DType::U1, DType::U1], OpAttrs::None) => {
            crate::add_kernel!(Inplace, "add", U1, add_inplace_u1, add_inplace_u1_broadcast)
        }
        _ => None,
    }
}
