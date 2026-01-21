use crate::graph::OpAttrs;
use crate::ops::registry::InplaceKernelFn;
use crate::tensor::DType;

pub fn lookup_kernel_cpu_avx_matmul_inplace(
    _output_dtype: DType,
    _input_dtypes: &[DType],
    _attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    None
}
