use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::InplaceKernelFn;
use crate::tensor::{DType, TensorValue};

use super::{
    fill_inplace_bf16,
    fill_inplace_bitset,
    fill_inplace_bool,
    fill_inplace_f16,
    fill_inplace_f32,
    fill_inplace_f64,
    fill_inplace_f8,
    fill_inplace_i16,
    fill_inplace_i1,
    fill_inplace_i2,
    fill_inplace_i32,
    fill_inplace_i4,
    fill_inplace_i64,
    fill_inplace_i8,
    fill_inplace_u16,
    fill_inplace_u1,
    fill_inplace_u2,
    fill_inplace_u32,
    fill_inplace_u4,
    fill_inplace_u64,
    fill_inplace_u8,
};

#[allow(dead_code)]
pub fn supports_fill_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!((output_dtype, input_dtypes, attrs), (_, [_], OpAttrs::Fill { .. }))
}

pub fn lookup_kernel_cpu_fill_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_fill_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", I8, fill_inplace_i8)
        }
        (DType::I16, [DType::I16], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", I16, fill_inplace_i16)
        }
        (DType::I32, [DType::I32], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", I32, fill_inplace_i32)
        }
        (DType::I64, [DType::I64], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", I64, fill_inplace_i64)
        }
        (DType::U8, [DType::U8], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", U8, fill_inplace_u8)
        }
        (DType::U16, [DType::U16], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", U16, fill_inplace_u16)
        }
        (DType::U32, [DType::U32], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", U32, fill_inplace_u32)
        }
        (DType::U64, [DType::U64], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", U64, fill_inplace_u64)
        }
        (DType::F16, [DType::F16], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", F16, fill_inplace_f16)
        }
        (DType::BF16, [DType::BF16], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", BF16, fill_inplace_bf16)
        }
        (DType::F8E5M2, [DType::F8E5M2], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", F8E5M2, fill_inplace_f8)
        }
        (DType::F32, [DType::F32], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", F32, fill_inplace_f32)
        }
        (DType::F64, [DType::F64], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", F64, fill_inplace_f64)
        }
        (DType::Bool, [DType::Bool], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", Bool, fill_inplace_bool)
        }
        (DType::Bitset, [DType::Bitset], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", Bitset, fill_inplace_bitset)
        }
        (DType::I4, [DType::I4], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", I4, fill_inplace_i4)
        }
        (DType::I2, [DType::I2], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", I2, fill_inplace_i2)
        }
        (DType::I1, [DType::I1], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", I1, fill_inplace_i1)
        }
        (DType::U4, [DType::U4], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", U4, fill_inplace_u4)
        }
        (DType::U2, [DType::U2], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", U2, fill_inplace_u2)
        }
        (DType::U1, [DType::U1], OpAttrs::Fill { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "fill", U1, fill_inplace_u1)
        }
        _ => None,
    }
}
