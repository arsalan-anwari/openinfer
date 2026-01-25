use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, TensorElement, TensorValue};

use super::{
    abs_i16_i32,
    abs_i16_i64,
    abs_i32_i64,
    abs_i8_i16,
    abs_i8_i32,
    abs_i8_i64,
    abs_i4_i8_packed,
    abs_i4_i16_packed,
    abs_i4_i32_packed,
    abs_i4_i64_packed,
    abs_i2_i8_packed,
    abs_i2_i16_packed,
    abs_i2_i32_packed,
    abs_i2_i64_packed,
    abs_i1_i8_packed,
    abs_i1_i16_packed,
    abs_i1_i32_packed,
    abs_i1_i64_packed,
};

pub fn lookup_kernel_cpu_abs_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I16, [DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", i8, I16, abs_i8_i16)
        }
        (DType::I32, [DType::I8], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", i8, I32, abs_i8_i32)
        }
        (DType::I64, [DType::I8], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", i8, I64, abs_i8_i64)
        }
        (DType::I32, [DType::I16], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", i16, I32, abs_i16_i32)
        }
        (DType::I64, [DType::I16], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", i16, I64, abs_i16_i64)
        }
        (DType::I64, [DType::I32], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", i32, I64, abs_i32_i64)
        }
        (DType::I8, [DType::I4], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", crate::tensor::I4, I8, abs_i4_i8_packed)
        }
        (DType::I16, [DType::I4], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", crate::tensor::I4, I16, abs_i4_i16_packed)
        }
        (DType::I32, [DType::I4], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", crate::tensor::I4, I32, abs_i4_i32_packed)
        }
        (DType::I64, [DType::I4], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", crate::tensor::I4, I64, abs_i4_i64_packed)
        }
        (DType::I8, [DType::I2], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", crate::tensor::I2, I8, abs_i2_i8_packed)
        }
        (DType::I16, [DType::I2], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", crate::tensor::I2, I16, abs_i2_i16_packed)
        }
        (DType::I32, [DType::I2], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", crate::tensor::I2, I32, abs_i2_i32_packed)
        }
        (DType::I64, [DType::I2], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", crate::tensor::I2, I64, abs_i2_i64_packed)
        }
        (DType::I8, [DType::I1], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", crate::tensor::I1, I8, abs_i1_i8_packed)
        }
        (DType::I16, [DType::I1], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", crate::tensor::I1, I16, abs_i1_i16_packed)
        }
        (DType::I32, [DType::I1], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", crate::tensor::I1, I32, abs_i1_i32_packed)
        }
        (DType::I64, [DType::I1], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(AccumulateUnary, "abs", crate::tensor::I1, I64, abs_i1_i64_packed)
        }
        _ => None,
    }
}
