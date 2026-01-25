use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::DType;

pub fn lookup_kernel_vulkan_is_finite_inplace(
    _output_dtype: DType,
    _input_dtypes: &[DType],
    _attrs: &OpAttrs,
) -> Option<KernelFn> {
    None
}
