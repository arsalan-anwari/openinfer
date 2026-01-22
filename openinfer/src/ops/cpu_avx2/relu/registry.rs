use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel, KernelFn};
use crate::tensor::DType;
use crate::tensor::{I4, Tensor, TensorElement, TensorOptions};

use super::{relu_f32, relu_i4};

pub fn lookup_kernel_cpu_avx2_relu(
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
        (DType::I4, [DType::I4], &OpAttrs::Relu { .. }) => {
            Some(KernelFn::Host(Box::new(|attrs, inputs, _output, thread_id| {
                let a = <I4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("relu input 0 dtype mismatch"))?;
                let out = relu_i4(attrs, &a.data, a.numel(), thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(a.shape().to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })?;
                Ok(Some(<I4 as TensorElement>::into_value(tensor)))
            })))
        }
        _ => None,
    }
}
