use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel_out, KernelFn};
use crate::tensor::DType;

use super::{mul_i8_i16, mul_u8_u16};

pub fn lookup_kernel_cpu_avx2_mul_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I16, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                mul_i8_i16 as fn(&[i8], &[i8], usize) -> anyhow::Result<Vec<i16>>,
            )))
        }
        (DType::U16, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                mul_u8_u16 as fn(&[u8], &[u8], usize) -> anyhow::Result<Vec<u16>>,
            )))
        }
        _ => None,
    }
}
