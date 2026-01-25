use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::DType;

use super::add_accumulate_generic;

pub fn lookup_kernel_vulkan_add_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (out, [a, b], &OpAttrs::Accumulate { dtype }) if *a == *b && out == dtype => {
            Some(KernelFn::Vulkan(Box::new(move |attrs, buffers, thread_id| {
                add_accumulate_generic(attrs, buffers[0], buffers[1], out, None, thread_id)
            })))
        }
        _ => None,
    }
}
