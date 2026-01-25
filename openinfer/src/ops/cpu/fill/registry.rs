use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, TensorElement, TensorValue};

use super::{
    fill_bf16,
    fill_bitset,
    fill_bool,
    fill_f16,
    fill_f32,
    fill_f64,
    fill_f8,
    fill_i16,
    fill_i1,
    fill_i2,
    fill_i32,
    fill_i4,
    fill_i64,
    fill_i8,
    fill_u16,
    fill_u1,
    fill_u2,
    fill_u32,
    fill_u4,
    fill_u64,
    fill_u8,
};

pub fn lookup_kernel_cpu_fill(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", i8, I8, fill_i8)
        }
        (DType::I16, [DType::I16], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", i16, I16, fill_i16)
        }
        (DType::I32, [DType::I32], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", i32, I32, fill_i32)
        }
        (DType::I64, [DType::I64], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", i64, I64, fill_i64)
        }
        (DType::U8, [DType::U8], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", u8, U8, fill_u8)
        }
        (DType::U16, [DType::U16], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", u16, U16, fill_u16)
        }
        (DType::U32, [DType::U32], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", u32, U32, fill_u32)
        }
        (DType::U64, [DType::U64], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", u64, U64, fill_u64)
        }
        (DType::F16, [DType::F16], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", crate::tensor::F16, F16, fill_f16)
        }
        (DType::BF16, [DType::BF16], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", crate::tensor::BF16, BF16, fill_bf16)
        }
        (DType::F8E5M2, [DType::F8E5M2], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", crate::tensor::F8E5M2, F8E5M2, fill_f8)
        }
        (DType::F32, [DType::F32], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", f32, F32, fill_f32)
        }
        (DType::F64, [DType::F64], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", f64, F64, fill_f64)
        }
        (DType::Bool, [DType::Bool], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", bool, Bool, fill_bool)
        }
        (DType::Bitset, [DType::Bitset], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", crate::tensor::Bitset, Bitset, fill_bitset)
        }
        (DType::I4, [DType::I4], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", crate::tensor::I4, I4, fill_i4)
        }
        (DType::I2, [DType::I2], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", crate::tensor::I2, I2, fill_i2)
        }
        (DType::I1, [DType::I1], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", crate::tensor::I1, I1, fill_i1)
        }
        (DType::U4, [DType::U4], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", crate::tensor::U4, U4, fill_u4)
        }
        (DType::U2, [DType::U2], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", crate::tensor::U2, U2, fill_u2)
        }
        (DType::U1, [DType::U1], &OpAttrs::Fill { .. }) => {
            crate::add_kernel!(UnaryAttrs, "fill", crate::tensor::U1, U1, fill_u1)
        }
        _ => None,
    }
}
