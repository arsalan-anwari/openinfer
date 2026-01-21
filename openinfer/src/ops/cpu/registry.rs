use crate::graph::{OpAttrs, OpKind};
use crate::ops::registry::KernelFn;
use crate::tensor::DType;

use super::{abs, add, fill, is_finite, matmul, mul, relu};

pub fn lookup_kernel_cpu(
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match op {
        OpKind::Add => match attrs {
            OpAttrs::Accumulate { .. } => add::registry_accumulate::lookup_kernel_cpu_add_accumulate(
                output_dtype,
                input_dtypes,
                attrs,
            ),
            _ => add::registry::lookup_kernel_cpu_add(output_dtype, input_dtypes, attrs),
        },
        OpKind::Mul => match attrs {
            OpAttrs::Accumulate { .. } => mul::registry_accumulate::lookup_kernel_cpu_mul_accumulate(
                output_dtype,
                input_dtypes,
                attrs,
            ),
            _ => mul::registry::lookup_kernel_cpu_mul(output_dtype, input_dtypes, attrs),
        },
        OpKind::Abs => match attrs {
            OpAttrs::Accumulate { .. } => abs::registry_accumulate::lookup_kernel_cpu_abs_accumulate(
                output_dtype,
                input_dtypes,
                attrs,
            ),
            _ => abs::registry::lookup_kernel_cpu_abs(output_dtype, input_dtypes, attrs),
        },
        OpKind::Relu => match attrs {
            OpAttrs::Accumulate { .. } => relu::registry_accumulate::lookup_kernel_cpu_relu_accumulate(
                output_dtype,
                input_dtypes,
                attrs,
            ),
            _ => relu::registry::lookup_kernel_cpu_relu(output_dtype, input_dtypes, attrs),
        },
        OpKind::Matmul => match attrs {
            OpAttrs::Accumulate { .. } => {
                matmul::registry_accumulate::lookup_kernel_cpu_matmul_accumulate(
                    output_dtype,
                    input_dtypes,
                    attrs,
                )
            }
            _ => matmul::registry::lookup_kernel_cpu_matmul(
                output_dtype,
                input_dtypes,
                attrs,
            ),
        },
        OpKind::IsFinite => match attrs {
            OpAttrs::Accumulate { .. } => {
                is_finite::registry_accumulate::lookup_kernel_cpu_is_finite_accumulate(
                    output_dtype,
                    input_dtypes,
                    attrs,
                )
            }
            _ => is_finite::registry::lookup_kernel_cpu_is_finite(
                output_dtype,
                input_dtypes,
                attrs,
            ),
        },
        OpKind::Fill => match attrs {
            OpAttrs::Accumulate { .. } => {
                fill::registry_accumulate::lookup_kernel_cpu_fill_accumulate(
                    output_dtype,
                    input_dtypes,
                    attrs,
                )
            }
            _ => fill::registry::lookup_kernel_cpu_fill(
                output_dtype,
                input_dtypes,
                attrs,
            ),
        },
    }
}
