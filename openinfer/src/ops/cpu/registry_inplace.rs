use crate::graph::{OpAttrs, OpKind};
use crate::ops::registry::InplaceKernelFn;
use crate::tensor::DType;

use super::{abs, add, fill, matmul, mul, relu};

pub fn lookup_kernel_cpu_inplace(
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    match op {
        OpKind::Add => add::registry_inplace::lookup_kernel_cpu_add_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Mul => mul::registry_inplace::lookup_kernel_cpu_mul_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Abs => abs::registry_inplace::lookup_kernel_cpu_abs_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Relu => relu::registry_inplace::lookup_kernel_cpu_relu_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Fill => fill::registry_inplace::lookup_kernel_cpu_fill_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Matmul => matmul::registry_inplace::lookup_kernel_cpu_matmul_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::IsFinite => None,
    }
}
