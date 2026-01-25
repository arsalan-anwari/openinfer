use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::KernelFn;
use crate::tensor::{DType, TensorElement, TensorValue};

use super::{is_finite_bf16, is_finite_f16, is_finite_f32, is_finite_f64, is_finite_f8};

pub fn lookup_kernel_cpu_is_finite(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::Bool, [DType::F8E5M2], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "is_finite", crate::tensor::F8E5M2, Bool, is_finite_f8)
        }
        (DType::Bool, [DType::BF16], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "is_finite", crate::tensor::BF16, Bool, is_finite_bf16)
        }
        (DType::Bool, [DType::F16], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "is_finite", crate::tensor::F16, Bool, is_finite_f16)
        }
        (DType::Bool, [DType::F32], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "is_finite", f32, Bool, is_finite_f32)
        }
        (DType::Bool, [DType::F64], &OpAttrs::None) => {
            crate::add_kernel!(Unary, "is_finite", f64, Bool, is_finite_f64)
        }
        _ => None,
    }
}
