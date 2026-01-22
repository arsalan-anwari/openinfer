use crate::graph::{OpAttrs, OpKind};
use crate::ops::registry::InplaceKernelFn;
use crate::tensor::DType;

use super::{abs, add, matmul, mul, relu};

pub fn lookup_kernel_cpu_avx2_inplace(
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    let kernel = match op {
        OpKind::Add => add::registry_inplace::lookup_kernel_cpu_avx2_add_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Mul => mul::registry_inplace::lookup_kernel_cpu_avx2_mul_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Abs => abs::registry_inplace::lookup_kernel_cpu_avx2_abs_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Relu => relu::registry_inplace::lookup_kernel_cpu_avx2_relu_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Fill => None,
        OpKind::Matmul => matmul::registry_inplace::lookup_kernel_cpu_avx2_matmul_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::IsFinite => None,
    };
    kernel.or_else(|| crate::ops::cpu::registry_inplace::lookup_kernel_cpu_inplace(op, output_dtype, input_dtypes, attrs))
}
