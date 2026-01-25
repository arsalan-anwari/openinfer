use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::InplaceKernelFn;
use crate::tensor::{DType, TensorValue};

use super::{
    relu_inplace_bf16,
    relu_inplace_f16,
    relu_inplace_f32,
    relu_inplace_f64,
    relu_inplace_f8,
    relu_inplace_i16,
    relu_inplace_i32,
    relu_inplace_i4,
    relu_inplace_i64,
    relu_inplace_i8,
};

#[allow(dead_code)]
pub fn supports_relu_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::F32, [DType::F32], OpAttrs::Relu { .. })
            | (DType::F64, [DType::F64], OpAttrs::Relu { .. })
            | (DType::F16, [DType::F16], OpAttrs::Relu { .. })
            | (DType::BF16, [DType::BF16], OpAttrs::Relu { .. })
            | (DType::F8E5M2, [DType::F8E5M2], OpAttrs::Relu { .. })
            | (DType::I8, [DType::I8], OpAttrs::Relu { .. })
            | (DType::I16, [DType::I16], OpAttrs::Relu { .. })
            | (DType::I32, [DType::I32], OpAttrs::Relu { .. })
            | (DType::I64, [DType::I64], OpAttrs::Relu { .. })
            | (DType::I4, [DType::I4], OpAttrs::Relu { .. })
    )
}

pub fn lookup_kernel_cpu_relu_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_relu_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    match (output_dtype, input_dtypes, attrs) {
        (DType::F32, [DType::F32], OpAttrs::Relu { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "relu", F32, relu_inplace_f32)
        }
        (DType::F64, [DType::F64], OpAttrs::Relu { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "relu", F64, relu_inplace_f64)
        }
        (DType::F16, [DType::F16], OpAttrs::Relu { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "relu", F16, relu_inplace_f16)
        }
        (DType::BF16, [DType::BF16], OpAttrs::Relu { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "relu", BF16, relu_inplace_bf16)
        }
        (DType::F8E5M2, [DType::F8E5M2], OpAttrs::Relu { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "relu", F8E5M2, relu_inplace_f8)
        }
        (DType::I8, [DType::I8], OpAttrs::Relu { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "relu", I8, relu_inplace_i8)
        }
        (DType::I16, [DType::I16], OpAttrs::Relu { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "relu", I16, relu_inplace_i16)
        }
        (DType::I32, [DType::I32], OpAttrs::Relu { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "relu", I32, relu_inplace_i32)
        }
        (DType::I64, [DType::I64], OpAttrs::Relu { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "relu", I64, relu_inplace_i64)
        }
        (DType::I4, [DType::I4], OpAttrs::Relu { .. }) => {
            crate::add_kernel!(InplaceUnaryAttrs, "relu", I4, relu_inplace_i4)
        }
        _ => None,
    }
}
