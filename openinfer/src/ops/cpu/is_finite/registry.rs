use crate::graph::OpAttrs;
use crate::ops::registry::KernelFn;
use crate::tensor::DType;

use super::is_finite_kernel;

pub fn lookup_kernel_cpu_is_finite(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::Bool, [DType::F8E5M2], &OpAttrs::None)
        | (DType::Bool, [DType::BF16], &OpAttrs::None)
        | (DType::Bool, [DType::F16], &OpAttrs::None)
        | (DType::Bool, [DType::F32], &OpAttrs::None)
        | (DType::Bool, [DType::F64], &OpAttrs::None) => Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
            is_finite_kernel(inputs, thread_id)
        }))),
        _ => None,
    }
}
