use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::DType;

pub fn lookup_kernel_cpu_avx2_matmul(
    _output_dtype: DType,
    _input_dtypes: &[DType],
    _attrs: &OpAttrs,
) -> Option<KernelFn> {
    None
}
