use crate::graph::OpAttrs;
use crate::ops::registry::KernelFn;
use crate::tensor::DType;

pub fn lookup_kernel_cpu_is_finite_accumulate(
    _output_dtype: DType,
    _input_dtypes: &[DType],
    _attrs: &OpAttrs,
) -> Option<KernelFn> {
    None
}
