use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::{HostInplaceKernel, InplaceKernelFn};
use crate::tensor::{DType, TensorValue};

use super::{
    mul_inplace_bf16, mul_inplace_bitset, mul_inplace_bool, mul_inplace_f16, mul_inplace_f32,
    mul_inplace_f64, mul_inplace_f8, mul_inplace_i16, mul_inplace_i32, mul_inplace_i64,
    mul_inplace_i8, mul_inplace_u16, mul_inplace_u32, mul_inplace_u64, mul_inplace_u8,
    mul_inplace_i4, mul_inplace_i2, mul_inplace_i1, mul_inplace_u4, mul_inplace_u2, mul_inplace_u1,
};

#[allow(dead_code)]
pub fn supports_mul_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::I8, [DType::I8, DType::I8], OpAttrs::None)
            | (DType::I16, [DType::I16, DType::I16], OpAttrs::None)
            | (DType::F32, [DType::F32, DType::F32], OpAttrs::None)
            | (DType::F64, [DType::F64, DType::F64], OpAttrs::None)
            | (DType::F16, [DType::F16, DType::F16], OpAttrs::None)
            | (DType::BF16, [DType::BF16, DType::BF16], OpAttrs::None)
            | (DType::F8E5M2, [DType::F8E5M2, DType::F8E5M2], OpAttrs::None)
            | (DType::U8, [DType::U8, DType::U8], OpAttrs::None)
            | (DType::U16, [DType::U16, DType::U16], OpAttrs::None)
            | (DType::I32, [DType::I32, DType::I32], OpAttrs::None)
            | (DType::I64, [DType::I64, DType::I64], OpAttrs::None)
            | (DType::U32, [DType::U32, DType::U32], OpAttrs::None)
            | (DType::U64, [DType::U64, DType::U64], OpAttrs::None)
            | (DType::Bool, [DType::Bool, DType::Bool], OpAttrs::None)
            | (DType::Bitset, [DType::Bitset, DType::Bitset], OpAttrs::None)
            | (DType::I4, [DType::I4, DType::I4], OpAttrs::None)
            | (DType::I2, [DType::I2, DType::I2], OpAttrs::None)
            | (DType::I1, [DType::I1, DType::I1], OpAttrs::None)
            | (DType::U4, [DType::U4, DType::U4], OpAttrs::None)
            | (DType::U2, [DType::U2, DType::U2], OpAttrs::None)
            | (DType::U1, [DType::U1, DType::U1], OpAttrs::None)
    )
}

pub fn lookup_kernel_cpu_mul_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_mul_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    let kernel: HostInplaceKernel = Box::new(|_attrs, output, inputs, thread_id| {
        let other = inputs
            .get(0)
            .ok_or_else(|| anyhow!("inplace mul expects at least 1 input"))?;
        match (output, other) {
            (TensorValue::I8(out), TensorValue::I8(b)) => mul_inplace_i8(&mut out.data, &b.data, thread_id),
            (TensorValue::I16(out), TensorValue::I16(b)) => mul_inplace_i16(&mut out.data, &b.data, thread_id),
            (TensorValue::F32(out), TensorValue::F32(b)) => mul_inplace_f32(&mut out.data, &b.data, thread_id),
            (TensorValue::F64(out), TensorValue::F64(b)) => mul_inplace_f64(&mut out.data, &b.data, thread_id),
            (TensorValue::F16(out), TensorValue::F16(b)) => mul_inplace_f16(&mut out.data, &b.data, thread_id),
            (TensorValue::BF16(out), TensorValue::BF16(b)) => mul_inplace_bf16(&mut out.data, &b.data, thread_id),
            (TensorValue::F8E5M2(out), TensorValue::F8E5M2(b)) => mul_inplace_f8(&mut out.data, &b.data, thread_id),
            (TensorValue::U8(out), TensorValue::U8(b)) => mul_inplace_u8(&mut out.data, &b.data, thread_id),
            (TensorValue::U16(out), TensorValue::U16(b)) => mul_inplace_u16(&mut out.data, &b.data, thread_id),
            (TensorValue::I32(out), TensorValue::I32(b)) => mul_inplace_i32(&mut out.data, &b.data, thread_id),
            (TensorValue::I64(out), TensorValue::I64(b)) => mul_inplace_i64(&mut out.data, &b.data, thread_id),
            (TensorValue::U32(out), TensorValue::U32(b)) => mul_inplace_u32(&mut out.data, &b.data, thread_id),
            (TensorValue::U64(out), TensorValue::U64(b)) => mul_inplace_u64(&mut out.data, &b.data, thread_id),
            (TensorValue::Bool(out), TensorValue::Bool(b)) => {
                mul_inplace_bool(&mut out.data, &b.data, thread_id)
            }
            (TensorValue::Bitset(out), TensorValue::Bitset(b)) => {
                mul_inplace_bitset(&mut out.data, &b.data, thread_id)
            }
            (TensorValue::I4(out), TensorValue::I4(b)) => {
                let len = out.numel();
                mul_inplace_i4(&mut out.data, &b.data, len, thread_id)
            }
            (TensorValue::I2(out), TensorValue::I2(b)) => {
                let len = out.numel();
                mul_inplace_i2(&mut out.data, &b.data, len, thread_id)
            }
            (TensorValue::I1(out), TensorValue::I1(b)) => {
                let len = out.numel();
                mul_inplace_i1(&mut out.data, &b.data, len, thread_id)
            }
            (TensorValue::U4(out), TensorValue::U4(b)) => {
                let len = out.numel();
                mul_inplace_u4(&mut out.data, &b.data, len, thread_id)
            }
            (TensorValue::U2(out), TensorValue::U2(b)) => {
                let len = out.numel();
                mul_inplace_u2(&mut out.data, &b.data, len, thread_id)
            }
            (TensorValue::U1(out), TensorValue::U1(b)) => {
                let len = out.numel();
                mul_inplace_u1(&mut out.data, &b.data, len, thread_id)
            }
            _ => Err(anyhow!("inplace mul dtype mismatch")),
        }
    });
    Some(InplaceKernelFn::Host(kernel))
}
