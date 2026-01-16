use crate::graph::{OpAttrs, OpKind};
use crate::ops::registry::KernelFn;
use crate::tensor::DType;

use super::{abs, add, mul, relu};
use crate::ops::cpu;

pub fn lookup_kernel_cpu_avx2(
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match op {
        OpKind::Add => add::registry::lookup_kernel_cpu_avx2_add(output_dtype, input_dtypes, attrs),
        OpKind::Mul => mul::registry::lookup_kernel_cpu_avx2_mul(output_dtype, input_dtypes, attrs),
        OpKind::Abs => abs::registry::lookup_kernel_cpu_avx2_abs(output_dtype, input_dtypes, attrs),
        OpKind::Relu => relu::registry::lookup_kernel_cpu_avx2_relu(output_dtype, input_dtypes, attrs),
        OpKind::Matmul => cpu::matmul::registry::lookup_kernel_cpu_matmul(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::IsFinite => cpu::is_finite::registry::lookup_kernel_cpu_is_finite(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Fill => cpu::fill::registry::lookup_kernel_cpu_fill(
            output_dtype,
            input_dtypes,
            attrs,
        ),
    }
}
