use anyhow::Result;

use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::ops::{device_kernel, KernelFn};
use crate::tensor::DType;

use super::abs_inplace_generic;

pub fn lookup_kernel_vulkan_abs_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (out, [a], &OpAttrs::None)
            if out == *a
                && matches!(
                    out,
                    DType::F32
                        | DType::F64
                        | DType::I8
                        | DType::I16
                        | DType::I32
                        | DType::I64
                        | DType::I4
                        | DType::I2
                        | DType::I1
                ) =>
        {
            Some(KernelFn::Vulkan(device_kernel(
                abs_inplace_generic as fn(&OpAttrs, &VulkanBuffer, usize) -> Result<VulkanBuffer>,
            )))
        }
        _ => None,
    }
}
