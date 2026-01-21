use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel, KernelFn};
use crate::tensor::{DType, I4, Tensor, TensorElement, TensorOptions};

use super::{
    relu_bf16, relu_f16, relu_f32, relu_f64, relu_f8, relu_i16, relu_i32, relu_i64, relu_i8,
    relu_i4,
};

pub fn lookup_kernel_cpu_relu(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::F32, [DType::F32], &OpAttrs::Relu { .. }) => {
            Some(KernelFn::Host(cpu_kernel(
                relu_f32 as fn(&OpAttrs, &[f32], usize) -> Result<Vec<f32>>,
            )))
        }
        (DType::F64, [DType::F64], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            relu_f64 as fn(&OpAttrs, &[f64], usize) -> Result<Vec<f64>>,
        ))),
        (DType::F16, [DType::F16], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            relu_f16 as fn(&OpAttrs, &[crate::tensor::F16], usize) -> Result<Vec<crate::tensor::F16>>,
        ))),
        (DType::BF16, [DType::BF16], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            relu_bf16 as fn(&OpAttrs, &[crate::tensor::BF16], usize) -> Result<Vec<crate::tensor::BF16>>,
        ))),
        (DType::F8E5M2, [DType::F8E5M2], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            relu_f8 as fn(&OpAttrs, &[crate::tensor::F8E5M2], usize) -> Result<Vec<crate::tensor::F8E5M2>>,
        ))),
        (DType::I8, [DType::I8], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            relu_i8 as fn(&OpAttrs, &[i8], usize) -> Result<Vec<i8>>,
        ))),
        (DType::I16, [DType::I16], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            relu_i16 as fn(&OpAttrs, &[i16], usize) -> Result<Vec<i16>>,
        ))),
        (DType::I32, [DType::I32], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            relu_i32 as fn(&OpAttrs, &[i32], usize) -> Result<Vec<i32>>,
        ))),
        (DType::I64, [DType::I64], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(cpu_kernel(
            relu_i64 as fn(&OpAttrs, &[i64], usize) -> Result<Vec<i64>>,
        ))),
        (DType::I4, [DType::I4], &OpAttrs::Relu { .. }) => Some(KernelFn::Host(Box::new(|attrs, inputs, thread_id| {
            let a = <I4 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("relu input 0 dtype mismatch"))?;
            let out = relu_i4(attrs, &a.data, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(<I4 as TensorElement>::into_value(tensor))
        }))),
        _ => None,
    }
}
