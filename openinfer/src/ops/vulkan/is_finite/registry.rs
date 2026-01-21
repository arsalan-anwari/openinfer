use anyhow::Result;

use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::ops::{device_kernel, KernelFn};
use crate::tensor::DType;

use super::is_finite_generic;

pub fn lookup_kernel_vulkan_is_finite(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::Bool, [input], &OpAttrs::None)
            if matches!(
                input,
                DType::F32 | DType::F64
            ) =>
        {
            Some(KernelFn::Vulkan(device_kernel(
                is_finite_generic as fn(&OpAttrs, &VulkanBuffer, usize) -> Result<VulkanBuffer>,
            )))
        }
        _ => None,
    }
}
