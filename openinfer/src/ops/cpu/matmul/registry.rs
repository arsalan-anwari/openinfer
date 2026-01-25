use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::KernelFn;
use crate::tensor::{DType, TensorElement, TensorValue};

use super::{
    matmul_bitset,
    matmul_bool,
    matmul_f16,
    matmul_f32,
    matmul_f64,
    matmul_i16,
    matmul_i32,
    matmul_i64,
    matmul_i8,
    matmul_u16,
    matmul_u32,
    matmul_u64,
    matmul_u8,
};

pub fn lookup_kernel_cpu_matmul(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", i8, I8, matmul_i8)
        }
        (DType::I16, [DType::I16, DType::I16], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", i16, I16, matmul_i16)
        }
        (DType::I32, [DType::I32, DType::I32], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", i32, I32, matmul_i32)
        }
        (DType::I64, [DType::I64, DType::I64], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", i64, I64, matmul_i64)
        }
        (DType::U8, [DType::U8, DType::U8], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", u8, U8, matmul_u8)
        }
        (DType::U16, [DType::U16, DType::U16], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", u16, U16, matmul_u16)
        }
        (DType::U32, [DType::U32, DType::U32], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", u32, U32, matmul_u32)
        }
        (DType::U64, [DType::U64, DType::U64], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", u64, U64, matmul_u64)
        }
        (DType::F16, [DType::F16, DType::F16], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", crate::tensor::F16, F16, matmul_f16)
        }
        (DType::F32, [DType::F32, DType::F32], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", f32, F32, matmul_f32)
        }
        (DType::F64, [DType::F64, DType::F64], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", f64, F64, matmul_f64)
        }
        (DType::Bool, [DType::Bool, DType::Bool], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", bool, Bool, matmul_bool)
        }
        (DType::Bitset, [DType::Bitset, DType::Bitset], &OpAttrs::None) => {
            crate::add_kernel!(BinaryNoBroadcast, "matmul", crate::tensor::Bitset, Bitset, matmul_bitset)
        }
        _ => None,
    }
}
