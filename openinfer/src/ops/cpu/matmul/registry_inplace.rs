use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::{HostInplaceKernel, InplaceKernelFn};
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
    let kernel: HostInplaceKernel = Box::new(|_attrs, output, inputs, thread_id| {
        if inputs.len() < 1 {
            return Err(anyhow!("inplace matmul expects at least 1 input"));
        }
        let other = inputs
            .get(0)
            .ok_or_else(|| anyhow!("inplace matmul expects at least 1 input"))?;
        match (output, other) {
            (TensorValue::I8(out), TensorValue::I8(b)) => {
                matmul_inplace_i8(out, b, thread_id)
            }
            (TensorValue::I16(out), TensorValue::I16(b)) => {
                matmul_inplace_i16(out, b, thread_id)
            }
            (TensorValue::I32(out), TensorValue::I32(b)) => {
                matmul_inplace_i32(out, b, thread_id)
            }
            (TensorValue::I64(out), TensorValue::I64(b)) => {
                matmul_inplace_i64(out, b, thread_id)
            }
            (TensorValue::U8(out), TensorValue::U8(b)) => {
                matmul_inplace_u8(out, b, thread_id)
            }
            (TensorValue::U16(out), TensorValue::U16(b)) => {
                matmul_inplace_u16(out, b, thread_id)
            }
            (TensorValue::U32(out), TensorValue::U32(b)) => {
                matmul_inplace_u32(out, b, thread_id)
            }
            (TensorValue::U64(out), TensorValue::U64(b)) => {
                matmul_inplace_u64(out, b, thread_id)
            }
            (TensorValue::F16(out), TensorValue::F16(b)) => {
                matmul_inplace_f16(out, b, thread_id)
            }
            (TensorValue::F32(out), TensorValue::F32(b)) => {
                matmul_inplace_f32(out, b, thread_id)
            }
            (TensorValue::F64(out), TensorValue::F64(b)) => {
                matmul_inplace_f64(out, b, thread_id)
            }
            (TensorValue::Bool(out), TensorValue::Bool(b)) => {
                matmul_inplace_bool(out, b, thread_id)
            }
            (TensorValue::Bitset(out), TensorValue::Bitset(b)) => {
                matmul_inplace_bitset(out, b, thread_id)
            }
            _ => Err(anyhow!("inplace matmul dtype mismatch")),
        }
    });
    Some(InplaceKernelFn::Host(kernel))
}
