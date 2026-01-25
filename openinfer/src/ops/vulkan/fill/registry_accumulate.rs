use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::DType;

#[allow(dead_code)]
pub fn lookup_kernel_vulkan_fill_accumulate(
    _output_dtype: DType,
    _input_dtypes: &[DType],
    _attrs: &OpAttrs,
) -> Option<KernelFn> {
    None
}
