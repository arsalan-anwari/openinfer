use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, TensorElement, TensorValue};

use super::{
    add_bf16,
    add_bf16_broadcast,
    add_bitset,
    add_bitset_broadcast,
    add_bool,
    add_bool_broadcast,
    add_f16,
    add_f16_broadcast,
    add_f32,
    add_f32_broadcast,
    add_f64,
    add_f64_broadcast,
    add_f8,
    add_f8_broadcast,
    add_i16,
    add_i16_broadcast,
    add_i1,
    add_i1_broadcast,
    add_i2,
    add_i2_broadcast,
    add_i32,
    add_i32_broadcast,
    add_i4,
    add_i4_broadcast,
    add_i64,
    add_i64_broadcast,
    add_i8,
    add_i8_broadcast,
    add_u16,
    add_u16_broadcast,
    add_u1,
    add_u1_broadcast,
    add_u2,
    add_u2_broadcast,
    add_u32,
    add_u32_broadcast,
    add_u4,
    add_u4_broadcast,
    add_u64,
    add_u64_broadcast,
    add_u8,
    add_u8_broadcast,
};

pub fn lookup_kernel_cpu_add(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", i8, I8, add_i8, add_i8_broadcast)
        }
        (DType::I16, [DType::I16, DType::I16], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", i16, I16, add_i16, add_i16_broadcast)
        }
        (DType::F32, [DType::F32, DType::F32], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", f32, F32, add_f32, add_f32_broadcast)
        }
        (DType::F64, [DType::F64, DType::F64], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", f64, F64, add_f64, add_f64_broadcast)
        }
        (DType::F16, [DType::F16, DType::F16], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", crate::tensor::F16, F16, add_f16, add_f16_broadcast)
        }
        (DType::BF16, [DType::BF16, DType::BF16], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", crate::tensor::BF16, BF16, add_bf16, add_bf16_broadcast)
        }
        (DType::F8E5M2, [DType::F8E5M2, DType::F8E5M2], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", crate::tensor::F8E5M2, F8E5M2, add_f8, add_f8_broadcast)
        }
        (DType::U8, [DType::U8, DType::U8], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", u8, U8, add_u8, add_u8_broadcast)
        }
        (DType::U16, [DType::U16, DType::U16], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", u16, U16, add_u16, add_u16_broadcast)
        }
        (DType::I32, [DType::I32, DType::I32], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", i32, I32, add_i32, add_i32_broadcast)
        }
        (DType::I64, [DType::I64, DType::I64], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", i64, I64, add_i64, add_i64_broadcast)
        }
        (DType::U32, [DType::U32, DType::U32], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", u32, U32, add_u32, add_u32_broadcast)
        }
        (DType::U64, [DType::U64, DType::U64], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", u64, U64, add_u64, add_u64_broadcast)
        }
        (DType::Bool, [DType::Bool, DType::Bool], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", bool, Bool, add_bool, add_bool_broadcast)
        }
        (DType::Bitset, [DType::Bitset, DType::Bitset], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", crate::tensor::Bitset, Bitset, add_bitset, add_bitset_broadcast)
        }
        (DType::I4, [DType::I4, DType::I4], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", crate::tensor::I4, I4, add_i4, add_i4_broadcast)
        }
        (DType::I2, [DType::I2, DType::I2], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", crate::tensor::I2, I2, add_i2, add_i2_broadcast)
        }
        (DType::I1, [DType::I1, DType::I1], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", crate::tensor::I1, I1, add_i1, add_i1_broadcast)
        }
        (DType::U4, [DType::U4, DType::U4], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", crate::tensor::U4, U4, add_u4, add_u4_broadcast)
        }
        (DType::U2, [DType::U2, DType::U2], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", crate::tensor::U2, U2, add_u2, add_u2_broadcast)
        }
        (DType::U1, [DType::U1, DType::U1], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "add", crate::tensor::U1, U1, add_u1, add_u1_broadcast)
        }
        _ => None,
    }
}
