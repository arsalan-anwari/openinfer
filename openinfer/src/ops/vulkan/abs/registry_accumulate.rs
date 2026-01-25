use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::DType;

use super::abs_accumulate_generic;

pub fn lookup_kernel_vulkan_abs_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (out, [_], &OpAttrs::Accumulate { dtype }) if out == dtype => {
            Some(KernelFn::Vulkan(Box::new(move |attrs, buffers, thread_id| {
                abs_accumulate_generic(attrs, buffers[0], out, None, thread_id)
            })))
        }
        _ => None,
    }
}
