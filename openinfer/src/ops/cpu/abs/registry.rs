use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, TensorElement, TensorValue};

use super::{
    abs_bf16,
    abs_f16,
    abs_f32,
    abs_f64,
    abs_f8,
    abs_i16,
    abs_i1,
    abs_i2,
    abs_i32,
    abs_i4,
    abs_i64,
    abs_i8,
};

pub fn lookup_kernel_cpu_abs(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "abs", i8, I8, abs_i8)
        }
        (DType::I16, [DType::I16], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "abs", i16, I16, abs_i16)
        }
        (DType::F32, [DType::F32], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "abs", f32, F32, abs_f32)
        }
        (DType::F64, [DType::F64], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "abs", f64, F64, abs_f64)
        }
        (DType::F16, [DType::F16], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "abs", crate::tensor::F16, F16, abs_f16)
        }
        (DType::BF16, [DType::BF16], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "abs", crate::tensor::BF16, BF16, abs_bf16)
        }
        (DType::F8E5M2, [DType::F8E5M2], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "abs", crate::tensor::F8E5M2, F8E5M2, abs_f8)
        }
        (DType::I32, [DType::I32], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "abs", i32, I32, abs_i32)
        }
        (DType::I64, [DType::I64], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "abs", i64, I64, abs_i64)
        }
        (DType::I4, [DType::I4], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "abs", crate::tensor::I4, I4, abs_i4)
        }
        (DType::I2, [DType::I2], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "abs", crate::tensor::I2, I2, abs_i2)
        }
        (DType::I1, [DType::I1], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "abs", crate::tensor::I1, I1, abs_i1)
        }
        _ => None,
    }
}
