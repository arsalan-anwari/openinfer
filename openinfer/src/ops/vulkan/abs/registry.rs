use anyhow::Result;

use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::ops::{device_kernel, KernelFn};
use crate::tensor::DType;

use super::abs_generic;

pub fn lookup_kernel_vulkan_abs(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (out, [a], OpAttrs::None)
            if out == *a
                && matches!(
                    out,
                    DType::F32
                        | DType::I8
                        | DType::I16
                        | DType::I32
                        | DType::I64
                        | DType::U8
                        | DType::U16
                        | DType::U32
                        | DType::U64
                        | DType::Bool
                ) =>
        {
            Some(KernelFn::Vulkan(device_kernel(
                abs_generic as fn(&VulkanBuffer) -> Result<VulkanBuffer>,
            )))
        }
        _ => None,
    }
}
