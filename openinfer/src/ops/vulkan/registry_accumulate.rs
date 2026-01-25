use crate::graph::{OpAttrs, OpKind};
use crate::ops::registry::KernelFn;
use crate::tensor::DType;

use super::{abs, add, matmul, mul};

pub fn lookup_kernel_vulkan_accumulate(
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match op {
        OpKind::Add => add::registry_accumulate::lookup_kernel_vulkan_add_accumulate(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Mul => mul::registry_accumulate::lookup_kernel_vulkan_mul_accumulate(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Abs => abs::registry_accumulate::lookup_kernel_vulkan_abs_accumulate(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Matmul => matmul::registry_accumulate::lookup_kernel_vulkan_matmul_accumulate(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        _ => None,
    }
}
