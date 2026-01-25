use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::InplaceKernelFn;
use crate::tensor::{DType, TensorValue};

use super::{
    abs_inplace_f32, abs_inplace_f64, abs_inplace_i16, abs_inplace_i32, abs_inplace_i64,
    abs_inplace_i8, abs_inplace_i4, abs_inplace_i2,
};

pub fn supports_abs_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::I8, [DType::I8], OpAttrs::None)
            | (DType::I16, [DType::I16], OpAttrs::None)
            | (DType::F32, [DType::F32], OpAttrs::None)
            | (DType::F64, [DType::F64], OpAttrs::None)
            | (DType::I32, [DType::I32], OpAttrs::None)
            | (DType::I64, [DType::I64], OpAttrs::None)
            | (DType::I4, [DType::I4], OpAttrs::None)
            | (DType::I2, [DType::I2], OpAttrs::None)
    )
}

pub fn lookup_kernel_cpu_avx2_abs_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_abs_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8], OpAttrs::None) => {
            crate::add_kernel!(InplaceUnary, "abs", I8, abs_inplace_i8)
        }
        (DType::I16, [DType::I16], OpAttrs::None) => {
            crate::add_kernel!(InplaceUnary, "abs", I16, abs_inplace_i16)
        }
        (DType::F32, [DType::F32], OpAttrs::None) => {
            crate::add_kernel!(InplaceUnary, "abs", F32, abs_inplace_f32)
        }
        (DType::F64, [DType::F64], OpAttrs::None) => {
            crate::add_kernel!(InplaceUnary, "abs", F64, abs_inplace_f64)
        }
        (DType::I32, [DType::I32], OpAttrs::None) => {
            crate::add_kernel!(InplaceUnary, "abs", I32, abs_inplace_i32)
        }
        (DType::I64, [DType::I64], OpAttrs::None) => {
            crate::add_kernel!(InplaceUnary, "abs", I64, abs_inplace_i64)
        }
        (DType::I4, [DType::I4], OpAttrs::None) => {
            crate::add_kernel!(InplaceUnary, "abs", I4, abs_inplace_i4)
        }
        (DType::I2, [DType::I2], OpAttrs::None) => {
            crate::add_kernel!(InplaceUnary, "abs", I2, abs_inplace_i2)
        }
        _ => None,
    }
}
