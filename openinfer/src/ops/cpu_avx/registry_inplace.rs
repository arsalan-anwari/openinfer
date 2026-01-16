use crate::graph::{OpAttrs, OpKind};
use crate::ops::registry::InplaceKernelFn;
use crate::tensor::DType;

use super::{abs, add, mul, relu};

pub fn lookup_kernel_cpu_avx_inplace(
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    match op {
        OpKind::Add => add::registry_inplace::lookup_kernel_cpu_avx_add_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Mul => mul::registry_inplace::lookup_kernel_cpu_avx_mul_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Abs => abs::registry_inplace::lookup_kernel_cpu_avx_abs_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
        OpKind::Relu => relu::registry_inplace::lookup_kernel_cpu_avx_relu_inplace(
            output_dtype,
            input_dtypes,
            attrs,
        ),
    }
}
