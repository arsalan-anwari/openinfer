use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::{HostInplaceKernel, InplaceKernelFn};
use crate::tensor::{DType, TensorValue};

use super::{
    abs_inplace_bool, abs_inplace_f32, abs_inplace_f64, abs_inplace_i16, abs_inplace_i32,
    abs_inplace_i64, abs_inplace_i8, abs_inplace_u16, abs_inplace_u32, abs_inplace_u64,
    abs_inplace_u8,
};

pub fn supports_abs_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::I8, [DType::I8], OpAttrs::None)
            | (DType::I16, [DType::I16], OpAttrs::None)
            | (DType::F32, [DType::F32], OpAttrs::None)
            | (DType::F64, [DType::F64], OpAttrs::None)
            | (DType::I32, [DType::I32], OpAttrs::None)
            | (DType::I64, [DType::I64], OpAttrs::None)
            | (DType::U8, [DType::U8], OpAttrs::None)
            | (DType::U16, [DType::U16], OpAttrs::None)
            | (DType::U32, [DType::U32], OpAttrs::None)
            | (DType::U64, [DType::U64], OpAttrs::None)
            | (DType::Bool, [DType::Bool], OpAttrs::None)
    )
}

pub fn lookup_kernel_cpu_avx_abs_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_abs_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    let kernel: HostInplaceKernel = Box::new(|_attrs, output, _inputs, thread_id| {
        match output {
            TensorValue::I8(out) => abs_inplace_i8(&mut out.data, thread_id),
            TensorValue::I16(out) => abs_inplace_i16(&mut out.data, thread_id),
            TensorValue::F32(out) => abs_inplace_f32(&mut out.data, thread_id),
            TensorValue::F64(out) => abs_inplace_f64(&mut out.data, thread_id),
            TensorValue::U8(out) => abs_inplace_u8(&mut out.data, thread_id),
            TensorValue::U16(out) => abs_inplace_u16(&mut out.data, thread_id),
            TensorValue::I32(out) => abs_inplace_i32(&mut out.data, thread_id),
            TensorValue::I64(out) => abs_inplace_i64(&mut out.data, thread_id),
            TensorValue::U32(out) => abs_inplace_u32(&mut out.data, thread_id),
            TensorValue::U64(out) => abs_inplace_u64(&mut out.data, thread_id),
            TensorValue::Bool(out) => abs_inplace_bool(&mut out.data, thread_id),
            _ => Err(anyhow!("inplace abs dtype mismatch")),
        }
    });
    Some(InplaceKernelFn::Host(kernel))
}
