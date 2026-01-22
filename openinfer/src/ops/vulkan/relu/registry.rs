use anyhow::Result;

use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::ops::{device_kernel, KernelFn};
use crate::tensor::DType;

use super::relu_generic;

pub fn lookup_kernel_vulkan_relu(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (out, [a], &OpAttrs::Relu { .. })
            if out == *a
                && matches!(
                    out,
                    DType::F32
                        | DType::F16
                        | DType::BF16
                        | DType::F8E5M2
                        | DType::F64
                        | DType::I8
                        | DType::I16
                        | DType::I32
                        | DType::I64
                        | DType::I4
                ) =>
        {
            Some(KernelFn::Vulkan(device_kernel(
                relu_generic as fn(&OpAttrs, &VulkanBuffer, usize) -> Result<VulkanBuffer>,
            )))
        }
        _ => None,
    }
}
