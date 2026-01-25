use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, TensorElement, TensorValue};

use super::{
    matmul_i16_i32,
    matmul_i16_i64,
    matmul_i32_i64,
    matmul_i8_i16,
    matmul_i8_i32,
    matmul_i8_i64,
    matmul_u16_u32,
    matmul_u16_u64,
    matmul_u32_u64,
    matmul_u8_u16,
    matmul_u8_u32,
    matmul_u8_u64,
};

pub fn lookup_kernel_cpu_matmul_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I16, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(AccumulateBinaryNoBroadcast, "matmul", i8, I16, matmul_i8_i16)
        }
        (DType::I32, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(AccumulateBinaryNoBroadcast, "matmul", i8, I32, matmul_i8_i32)
        }
        (DType::I64, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(AccumulateBinaryNoBroadcast, "matmul", i8, I64, matmul_i8_i64)
        }
        (DType::I32, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(AccumulateBinaryNoBroadcast, "matmul", i16, I32, matmul_i16_i32)
        }
        (DType::I64, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(AccumulateBinaryNoBroadcast, "matmul", i16, I64, matmul_i16_i64)
        }
        (DType::I64, [DType::I32, DType::I32], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(AccumulateBinaryNoBroadcast, "matmul", i32, I64, matmul_i32_i64)
        }
        (DType::U16, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            crate::add_kernel!(AccumulateBinaryNoBroadcast, "matmul", u8, U16, matmul_u8_u16)
        }
        (DType::U32, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            crate::add_kernel!(AccumulateBinaryNoBroadcast, "matmul", u8, U32, matmul_u8_u32)
        }
        (DType::U64, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(AccumulateBinaryNoBroadcast, "matmul", u8, U64, matmul_u8_u64)
        }
        (DType::U32, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            crate::add_kernel!(AccumulateBinaryNoBroadcast, "matmul", u16, U32, matmul_u16_u32)
        }
        (DType::U64, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(AccumulateBinaryNoBroadcast, "matmul", u16, U64, matmul_u16_u64)
        }
        (DType::U64, [DType::U32, DType::U32], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(AccumulateBinaryNoBroadcast, "matmul", u32, U64, matmul_u32_u64)
        }
        _ => None,
    }
}
