use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::InplaceKernelFn;
use crate::tensor::{DType, TensorValue};

use super::{
    matmul_inplace_bitset, matmul_inplace_bool, matmul_inplace_f16, matmul_inplace_f32,
    matmul_inplace_f64, matmul_inplace_i16, matmul_inplace_i32, matmul_inplace_i64,
    matmul_inplace_i8, matmul_inplace_u16, matmul_inplace_u32, matmul_inplace_u64,
    matmul_inplace_u8,
};

#[allow(dead_code)]
pub fn supports_matmul_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::I8, [DType::I8, DType::I8], OpAttrs::None)
            | (DType::I16, [DType::I16, DType::I16], OpAttrs::None)
            | (DType::I32, [DType::I32, DType::I32], OpAttrs::None)
            | (DType::I64, [DType::I64, DType::I64], OpAttrs::None)
            | (DType::U8, [DType::U8, DType::U8], OpAttrs::None)
            | (DType::U16, [DType::U16, DType::U16], OpAttrs::None)
            | (DType::U32, [DType::U32, DType::U32], OpAttrs::None)
            | (DType::U64, [DType::U64, DType::U64], OpAttrs::None)
            | (DType::F16, [DType::F16, DType::F16], OpAttrs::None)
            | (DType::F32, [DType::F32, DType::F32], OpAttrs::None)
            | (DType::F64, [DType::F64, DType::F64], OpAttrs::None)
            | (DType::Bool, [DType::Bool, DType::Bool], OpAttrs::None)
            | (DType::Bitset, [DType::Bitset, DType::Bitset], OpAttrs::None)
    )
}

pub fn lookup_kernel_cpu_matmul_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_matmul_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", I8, matmul_inplace_i8)
        }
        (DType::I16, [DType::I16, DType::I16], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", I16, matmul_inplace_i16)
        }
        (DType::I32, [DType::I32, DType::I32], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", I32, matmul_inplace_i32)
        }
        (DType::I64, [DType::I64, DType::I64], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", I64, matmul_inplace_i64)
        }
        (DType::U8, [DType::U8, DType::U8], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", U8, matmul_inplace_u8)
        }
        (DType::U16, [DType::U16, DType::U16], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", U16, matmul_inplace_u16)
        }
        (DType::U32, [DType::U32, DType::U32], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", U32, matmul_inplace_u32)
        }
        (DType::U64, [DType::U64, DType::U64], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", U64, matmul_inplace_u64)
        }
        (DType::F16, [DType::F16, DType::F16], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", F16, matmul_inplace_f16)
        }
        (DType::F32, [DType::F32, DType::F32], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", F32, matmul_inplace_f32)
        }
        (DType::F64, [DType::F64, DType::F64], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", F64, matmul_inplace_f64)
        }
        (DType::Bool, [DType::Bool, DType::Bool], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", Bool, matmul_inplace_bool)
        }
        (DType::Bitset, [DType::Bitset, DType::Bitset], OpAttrs::None) => {
            crate::add_kernel!(InplaceBinaryNoBroadcast, "matmul", Bitset, matmul_inplace_bitset)
        }
        _ => None,
    }
}
