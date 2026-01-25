use crate::graph::{OpAttrs, OpKind};
use crate::ops::registry::KernelFn;
use crate::tensor::DType;

use super::{abs, add, fill, is_finite, matmul, mul, relu};

pub fn lookup_kernel_cpu_accumulate(
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match op {
        OpKind::Add => add::registry_accumulate::lookup_kernel_cpu_add_accumulate(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Mul => mul::registry_accumulate::lookup_kernel_cpu_mul_accumulate(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Abs => abs::registry_accumulate::lookup_kernel_cpu_abs_accumulate(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Relu => relu::registry_accumulate::lookup_kernel_cpu_relu_accumulate(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Matmul => matmul::registry_accumulate::lookup_kernel_cpu_matmul_accumulate(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::IsFinite => is_finite::registry_accumulate::lookup_kernel_cpu_is_finite_accumulate(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Fill => fill::registry_accumulate::lookup_kernel_cpu_fill_accumulate(
            output_dtype,
            input_dtypes,
            attrs,
        ),
    }
}
