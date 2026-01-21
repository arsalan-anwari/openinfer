use anyhow::Result;

use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::ops::{device_kernel, KernelFn};
use crate::tensor::DType;

use super::matmul_inplace_generic;

pub fn lookup_kernel_vulkan_matmul_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (out, [a, b], &OpAttrs::None)
            if out == *a
                && *a == *b
                && matches!(
                    out,
                    DType::F32
                        | DType::F64
                        | DType::I8
                        | DType::I16
                        | DType::I32
                        | DType::I64
                        | DType::U8
                        | DType::U16
                        | DType::U32
                        | DType::U64
                        | DType::Bool
                        | DType::Bitset
                        | DType::I4
                        | DType::I2
                        | DType::I1
                        | DType::U4
                        | DType::U2
                        | DType::U1
                ) =>
        {
            Some(KernelFn::Vulkan(device_kernel(
                matmul_inplace_generic
                    as fn(&OpAttrs, &VulkanBuffer, &VulkanBuffer, usize) -> Result<VulkanBuffer>,
            )))
        }
        _ => None,
    }
}
