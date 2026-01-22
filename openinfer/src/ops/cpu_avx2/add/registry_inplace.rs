use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::{HostInplaceKernel, InplaceKernelFn};
use crate::tensor::{DType, TensorValue};

use super::{
    add_inplace_bool, add_inplace_f32, add_inplace_f64, add_inplace_i16, add_inplace_i32,
    add_inplace_i64, add_inplace_i8, add_inplace_u16, add_inplace_u32, add_inplace_u64,
    add_inplace_u8, add_inplace_i4, add_inplace_i2, add_inplace_u4, add_inplace_u2,
};

pub fn supports_add_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::I8, [DType::I8, DType::I8], OpAttrs::None)
            | (DType::I16, [DType::I16, DType::I16], OpAttrs::None)
            | (DType::I32, [DType::I32, DType::I32], OpAttrs::None)
            | (DType::I64, [DType::I64, DType::I64], OpAttrs::None)
            | (DType::F32, [DType::F32, DType::F32], OpAttrs::None)
            | (DType::F64, [DType::F64, DType::F64], OpAttrs::None)
            | (DType::U8, [DType::U8, DType::U8], OpAttrs::None)
            | (DType::U16, [DType::U16, DType::U16], OpAttrs::None)
            | (DType::U32, [DType::U32, DType::U32], OpAttrs::None)
            | (DType::U64, [DType::U64, DType::U64], OpAttrs::None)
            | (DType::Bool, [DType::Bool, DType::Bool], OpAttrs::None)
            | (DType::I4, [DType::I4, DType::I4], OpAttrs::None)
            | (DType::I2, [DType::I2, DType::I2], OpAttrs::None)
            | (DType::U4, [DType::U4, DType::U4], OpAttrs::None)
            | (DType::U2, [DType::U2, DType::U2], OpAttrs::None)
    )
}

pub fn lookup_kernel_cpu_avx2_add_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_add_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    let kernel: HostInplaceKernel = Box::new(|_attrs, output, inputs, thread_id| {
        let other = inputs
            .get(0)
            .ok_or_else(|| anyhow!("inplace add expects at least 1 input"))?;
        match (output, other) {
            (TensorValue::I8(out), TensorValue::I8(b)) => add_inplace_i8(&mut out.data, &b.data, thread_id),
            (TensorValue::I16(out), TensorValue::I16(b)) => add_inplace_i16(&mut out.data, &b.data, thread_id),
            (TensorValue::I32(out), TensorValue::I32(b)) => add_inplace_i32(&mut out.data, &b.data, thread_id),
            (TensorValue::I64(out), TensorValue::I64(b)) => add_inplace_i64(&mut out.data, &b.data, thread_id),
            (TensorValue::F32(out), TensorValue::F32(b)) => add_inplace_f32(&mut out.data, &b.data, thread_id),
            (TensorValue::F64(out), TensorValue::F64(b)) => add_inplace_f64(&mut out.data, &b.data, thread_id),
            (TensorValue::U8(out), TensorValue::U8(b)) => add_inplace_u8(&mut out.data, &b.data, thread_id),
            (TensorValue::U16(out), TensorValue::U16(b)) => add_inplace_u16(&mut out.data, &b.data, thread_id),
            (TensorValue::U32(out), TensorValue::U32(b)) => add_inplace_u32(&mut out.data, &b.data, thread_id),
            (TensorValue::U64(out), TensorValue::U64(b)) => add_inplace_u64(&mut out.data, &b.data, thread_id),
            (TensorValue::Bool(out), TensorValue::Bool(b)) => add_inplace_bool(&mut out.data, &b.data, thread_id),
            (TensorValue::I4(out), TensorValue::I4(b)) => {
                let len = out.numel();
                add_inplace_i4(&mut out.data, &b.data, len, thread_id)
            }
            (TensorValue::I2(out), TensorValue::I2(b)) => {
                let len = out.numel();
                add_inplace_i2(&mut out.data, &b.data, len, thread_id)
            }
            (TensorValue::U4(out), TensorValue::U4(b)) => {
                let len = out.numel();
                add_inplace_u4(&mut out.data, &b.data, len, thread_id)
            }
            (TensorValue::U2(out), TensorValue::U2(b)) => {
                let len = out.numel();
                add_inplace_u2(&mut out.data, &b.data, len, thread_id)
            }
            _ => Err(anyhow!("inplace add dtype mismatch")),
        }
    });
    Some(InplaceKernelFn::Host(kernel))
}
