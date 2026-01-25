use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, TensorElement, TensorValue};

use super::{
    relu_bf16,
    relu_f16,
    relu_f32,
    relu_f64,
    relu_f8,
    relu_i16,
    relu_i32,
    relu_i4,
    relu_i64,
    relu_i8,
};

pub fn lookup_kernel_cpu_relu(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::F32, [DType::F32], &OpAttrs::Relu { .. }) => {
            crate::add_kernel!(UnaryAttrs, "relu", f32, F32, relu_f32)
        }
        (DType::F64, [DType::F64], &OpAttrs::Relu { .. }) => {
            crate::add_kernel!(UnaryAttrs, "relu", f64, F64, relu_f64)
        }
        (DType::F16, [DType::F16], &OpAttrs::Relu { .. }) => {
            crate::add_kernel!(UnaryAttrs, "relu", crate::tensor::F16, F16, relu_f16)
        }
        (DType::BF16, [DType::BF16], &OpAttrs::Relu { .. }) => {
            crate::add_kernel!(UnaryAttrs, "relu", crate::tensor::BF16, BF16, relu_bf16)
        }
        (DType::F8E5M2, [DType::F8E5M2], &OpAttrs::Relu { .. }) => {
            crate::add_kernel!(UnaryAttrs, "relu", crate::tensor::F8E5M2, F8E5M2, relu_f8)
        }
        (DType::I8, [DType::I8], &OpAttrs::Relu { .. }) => {
            crate::add_kernel!(UnaryAttrs, "relu", i8, I8, relu_i8)
        }
        (DType::I16, [DType::I16], &OpAttrs::Relu { .. }) => {
            crate::add_kernel!(UnaryAttrs, "relu", i16, I16, relu_i16)
        }
        (DType::I32, [DType::I32], &OpAttrs::Relu { .. }) => {
            crate::add_kernel!(UnaryAttrs, "relu", i32, I32, relu_i32)
        }
        (DType::I64, [DType::I64], &OpAttrs::Relu { .. }) => {
            crate::add_kernel!(UnaryAttrs, "relu", i64, I64, relu_i64)
        }
        (DType::I4, [DType::I4], &OpAttrs::Relu { .. }) => {
            crate::add_kernel!(UnaryAttrs, "relu", crate::tensor::I4, I4, relu_i4)
        }
        _ => None,
    }
}
