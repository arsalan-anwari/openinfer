use crate::graph::OpAttrs;
use crate::ops::registry::{HostInplaceKernel, InplaceKernelFn};
use crate::tensor::{DType, TensorValue};

use super::relu_inplace_f32;

pub fn supports_relu_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!((output_dtype, input_dtypes, attrs), (DType::F32, [DType::F32], OpAttrs::Relu { .. }))
}

pub fn lookup_kernel_cpu_avx2_relu_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_relu_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    let kernel: HostInplaceKernel = Box::new(|attrs, output, _inputs, thread_id| {
        match output {
            TensorValue::F32(out) => relu_inplace_f32(attrs, &mut out.data, thread_id),
            _ => Err(anyhow::anyhow!("relu inplace dtype mismatch")),
        }
    });
    Some(InplaceKernelFn::Host(kernel))
}
