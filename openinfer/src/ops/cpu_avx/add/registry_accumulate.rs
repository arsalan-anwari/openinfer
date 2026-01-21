use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel_out, KernelFn};
use crate::tensor::DType;

use super::{add_i16_i32, add_i32_i64, add_i8_i16, add_u16_u32, add_u32_u64, add_u8_u16};

pub fn lookup_kernel_cpu_avx_add_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I16, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_i8_i16 as fn(&[i8], &[i8], usize) -> anyhow::Result<Vec<i16>>,
            )))
        }
        (DType::I32, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_i16_i32 as fn(&[i16], &[i16], usize) -> anyhow::Result<Vec<i32>>,
            )))
        }
        (DType::I64, [DType::I32, DType::I32], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_i32_i64 as fn(&[i32], &[i32], usize) -> anyhow::Result<Vec<i64>>,
            )))
        }
        (DType::U16, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_u8_u16 as fn(&[u8], &[u8], usize) -> anyhow::Result<Vec<u16>>,
            )))
        }
        (DType::U32, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_u16_u32 as fn(&[u16], &[u16], usize) -> anyhow::Result<Vec<u32>>,
            )))
        }
        (DType::U64, [DType::U32, DType::U32], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_u32_u64 as fn(&[u32], &[u32], usize) -> anyhow::Result<Vec<u64>>,
            )))
        }
        _ => None,
    }
}
