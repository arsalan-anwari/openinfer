use anyhow::Result;

use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel, KernelFn};
use crate::tensor::DType;

use super::relu_f32;

pub fn lookup_kernel_cpu_avx_relu(
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
        _ => None,
    }
}
