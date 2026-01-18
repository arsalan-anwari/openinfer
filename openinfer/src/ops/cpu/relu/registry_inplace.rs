use anyhow::anyhow;
use crate::graph::OpAttrs;
use crate::ops::registry::{HostInplaceKernel, InplaceKernelFn};
use crate::tensor::{DType, TensorValue};

use super::{
    relu_inplace_bitset, relu_inplace_bool, relu_inplace_f16, relu_inplace_f32,
    relu_inplace_f64, relu_inplace_i16, relu_inplace_i32, relu_inplace_i64, relu_inplace_i8,
    relu_inplace_u16, relu_inplace_u32, relu_inplace_u64, relu_inplace_u8,
};

#[allow(dead_code)]
pub fn supports_relu_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::F32, [DType::F32], OpAttrs::Relu { .. })
            | (DType::F64, [DType::F64], OpAttrs::Relu { .. })
            | (DType::F16, [DType::F16], OpAttrs::Relu { .. })
            | (DType::I8, [DType::I8], OpAttrs::Relu { .. })
            | (DType::I16, [DType::I16], OpAttrs::Relu { .. })
            | (DType::I32, [DType::I32], OpAttrs::Relu { .. })
            | (DType::I64, [DType::I64], OpAttrs::Relu { .. })
            | (DType::U8, [DType::U8], OpAttrs::Relu { .. })
            | (DType::U16, [DType::U16], OpAttrs::Relu { .. })
            | (DType::U32, [DType::U32], OpAttrs::Relu { .. })
            | (DType::U64, [DType::U64], OpAttrs::Relu { .. })
            | (DType::Bool, [DType::Bool], OpAttrs::Relu { .. })
            | (DType::Bitset, [DType::Bitset], OpAttrs::Relu { .. })
    )
}

pub fn lookup_kernel_cpu_relu_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_relu_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    let kernel: HostInplaceKernel = Box::new(|attrs, output, _inputs, thread_id| {
        match output {
            TensorValue::F32(out) => relu_inplace_f32(attrs, &mut out.data, thread_id),
            TensorValue::F64(out) => relu_inplace_f64(attrs, &mut out.data, thread_id),
            TensorValue::F16(out) => relu_inplace_f16(attrs, &mut out.data, thread_id),
            TensorValue::I8(out) => relu_inplace_i8(attrs, &mut out.data, thread_id),
            TensorValue::I16(out) => relu_inplace_i16(attrs, &mut out.data, thread_id),
            TensorValue::I32(out) => relu_inplace_i32(attrs, &mut out.data, thread_id),
            TensorValue::I64(out) => relu_inplace_i64(attrs, &mut out.data, thread_id),
            TensorValue::U8(out) => relu_inplace_u8(attrs, &mut out.data, thread_id),
            TensorValue::U16(out) => relu_inplace_u16(attrs, &mut out.data, thread_id),
            TensorValue::U32(out) => relu_inplace_u32(attrs, &mut out.data, thread_id),
            TensorValue::U64(out) => relu_inplace_u64(attrs, &mut out.data, thread_id),
            TensorValue::Bool(out) => relu_inplace_bool(attrs, &mut out.data, thread_id),
            TensorValue::Bitset(out) => relu_inplace_bitset(attrs, &mut out.data, thread_id),
            other => Err(anyhow!("relu inplace does not support dtype {:?}", other.dtype())),
        }
    });
    Some(InplaceKernelFn::Host(kernel))
}
