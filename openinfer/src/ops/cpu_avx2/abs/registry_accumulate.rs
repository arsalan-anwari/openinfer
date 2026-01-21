use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel_out, KernelFn};
use crate::tensor::DType;

use super::{abs_i16_i32, abs_i8_i16};

pub fn lookup_kernel_cpu_avx2_abs_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I16, [DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                abs_i8_i16 as fn(&[i8], usize) -> anyhow::Result<Vec<i16>>,
            )))
        }
        (DType::I32, [DType::I16], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                abs_i16_i32 as fn(&[i16], usize) -> anyhow::Result<Vec<i32>>,
            )))
        }
        _ => None,
    }
}
