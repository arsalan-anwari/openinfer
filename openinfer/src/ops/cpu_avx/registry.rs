use crate::graph::{OpAttrs, OpKind};
use crate::ops::registry::KernelFn;
use crate::tensor::DType;

use super::{abs, add, matmul, mul, relu};

pub fn lookup_kernel_cpu_avx(
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match op {
        OpKind::Add => match attrs {
            OpAttrs::Accumulate { .. } => add::registry_accumulate::lookup_kernel_cpu_avx_add_accumulate(
                output_dtype,
                input_dtypes,
                attrs,
            ),
            _ => add::registry::lookup_kernel_cpu_avx_add(output_dtype, input_dtypes, attrs),
        },
        OpKind::Mul => match attrs {
            OpAttrs::Accumulate { .. } => mul::registry_accumulate::lookup_kernel_cpu_avx_mul_accumulate(
                output_dtype,
                input_dtypes,
                attrs,
            ),
            _ => mul::registry::lookup_kernel_cpu_avx_mul(output_dtype, input_dtypes, attrs),
        },
        OpKind::Abs => match attrs {
            OpAttrs::Accumulate { .. } => abs::registry_accumulate::lookup_kernel_cpu_avx_abs_accumulate(
                output_dtype,
                input_dtypes,
                attrs,
            ),
            _ => abs::registry::lookup_kernel_cpu_avx_abs(output_dtype, input_dtypes, attrs),
        },
        OpKind::Relu => match attrs {
            OpAttrs::Accumulate { .. } => {
                relu::registry_accumulate::lookup_kernel_cpu_avx_relu_accumulate(
                    output_dtype,
                    input_dtypes,
                    attrs,
                )
            }
            _ => relu::registry::lookup_kernel_cpu_avx_relu(output_dtype, input_dtypes, attrs),
        },
        OpKind::Matmul => match attrs {
            OpAttrs::Accumulate { .. } => {
                matmul::registry_accumulate::lookup_kernel_cpu_avx_matmul_accumulate(
                    output_dtype,
                    input_dtypes,
                    attrs,
                )
            }
            _ => matmul::registry::lookup_kernel_cpu_avx_matmul(output_dtype, input_dtypes, attrs),
        },
        OpKind::IsFinite => None,
        OpKind::Fill => None,
    }
}
