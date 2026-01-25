use crate::graph::{OpAttrs, OpKind};
use crate::ops::registry::KernelFn;
use crate::tensor::DType;

use super::{abs, add, fill, is_finite, matmul, mul, relu};

pub fn lookup_kernel_vulkan_inplace(
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match op {
        OpKind::Add => add::registry_inplace::lookup_kernel_vulkan_add_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Mul => mul::registry_inplace::lookup_kernel_vulkan_mul_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Abs => abs::registry_inplace::lookup_kernel_vulkan_abs_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Fill => fill::registry_inplace::lookup_kernel_vulkan_fill_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Matmul => matmul::registry_inplace::lookup_kernel_vulkan_matmul_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Relu => relu::registry_inplace::lookup_kernel_vulkan_relu_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::IsFinite => is_finite::registry_inplace::lookup_kernel_vulkan_is_finite_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
    }
}
