use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, TensorElement, TensorValue};

use super::{
    mul_bf16,
    mul_bf16_broadcast,
    mul_bitset,
    mul_bitset_broadcast,
    mul_bool,
    mul_bool_broadcast,
    mul_f16,
    mul_f16_broadcast,
    mul_f32,
    mul_f32_broadcast,
    mul_f64,
    mul_f64_broadcast,
    mul_f8,
    mul_f8_broadcast,
    mul_i16,
    mul_i16_broadcast,
    mul_i1,
    mul_i1_broadcast,
    mul_i2,
    mul_i2_broadcast,
    mul_i32,
    mul_i32_broadcast,
    mul_i4,
    mul_i4_broadcast,
    mul_i64,
    mul_i64_broadcast,
    mul_i8,
    mul_i8_broadcast,
    mul_u16,
    mul_u16_broadcast,
    mul_u1,
    mul_u1_broadcast,
    mul_u2,
    mul_u2_broadcast,
    mul_u32,
    mul_u32_broadcast,
    mul_u4,
    mul_u4_broadcast,
    mul_u64,
    mul_u64_broadcast,
    mul_u8,
    mul_u8_broadcast,
};

pub fn lookup_kernel_cpu_mul(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", i8, I8, mul_i8, mul_i8_broadcast)
        }
        (DType::I16, [DType::I16, DType::I16], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", i16, I16, mul_i16, mul_i16_broadcast)
        }
        (DType::F32, [DType::F32, DType::F32], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", f32, F32, mul_f32, mul_f32_broadcast)
        }
        (DType::F64, [DType::F64, DType::F64], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", f64, F64, mul_f64, mul_f64_broadcast)
        }
        (DType::F16, [DType::F16, DType::F16], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", crate::tensor::F16, F16, mul_f16, mul_f16_broadcast)
        }
        (DType::BF16, [DType::BF16, DType::BF16], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", crate::tensor::BF16, BF16, mul_bf16, mul_bf16_broadcast)
        }
        (DType::F8E5M2, [DType::F8E5M2, DType::F8E5M2], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", crate::tensor::F8E5M2, F8E5M2, mul_f8, mul_f8_broadcast)
        }
        (DType::U8, [DType::U8, DType::U8], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", u8, U8, mul_u8, mul_u8_broadcast)
        }
        (DType::U16, [DType::U16, DType::U16], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", u16, U16, mul_u16, mul_u16_broadcast)
        }
        (DType::I32, [DType::I32, DType::I32], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", i32, I32, mul_i32, mul_i32_broadcast)
        }
        (DType::I64, [DType::I64, DType::I64], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", i64, I64, mul_i64, mul_i64_broadcast)
        }
        (DType::U32, [DType::U32, DType::U32], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", u32, U32, mul_u32, mul_u32_broadcast)
        }
        (DType::U64, [DType::U64, DType::U64], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", u64, U64, mul_u64, mul_u64_broadcast)
        }
        (DType::Bool, [DType::Bool, DType::Bool], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", bool, Bool, mul_bool, mul_bool_broadcast)
        }
        (DType::Bitset, [DType::Bitset, DType::Bitset], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", crate::tensor::Bitset, Bitset, mul_bitset, mul_bitset_broadcast)
        }
        (DType::I4, [DType::I4, DType::I4], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", crate::tensor::I4, I4, mul_i4, mul_i4_broadcast)
        }
        (DType::I2, [DType::I2, DType::I2], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", crate::tensor::I2, I2, mul_i2, mul_i2_broadcast)
        }
        (DType::I1, [DType::I1, DType::I1], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", crate::tensor::I1, I1, mul_i1, mul_i1_broadcast)
        }
        (DType::U4, [DType::U4, DType::U4], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", crate::tensor::U4, U4, mul_u4, mul_u4_broadcast)
        }
        (DType::U2, [DType::U2, DType::U2], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", crate::tensor::U2, U2, mul_u2, mul_u2_broadcast)
        }
        (DType::U1, [DType::U1, DType::U1], &OpAttrs::None) => {
            crate::add_kernel!(Standard, "mul", crate::tensor::U1, U1, mul_u1, mul_u1_broadcast)
        }
        _ => None,
    }
}
