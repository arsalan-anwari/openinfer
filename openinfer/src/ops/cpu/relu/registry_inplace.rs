use anyhow::anyhow;
use crate::graph::OpAttrs;
use crate::ops::registry::{HostInplaceKernel, InplaceKernelFn};
use crate::tensor::{DType, TensorValue};

use super::{
    relu_inplace_bf16, relu_inplace_f16, relu_inplace_f32, relu_inplace_f64, relu_inplace_f8,
    relu_inplace_i16, relu_inplace_i32, relu_inplace_i64, relu_inplace_i8, relu_inplace_i4,
};

#[allow(dead_code)]
pub fn supports_relu_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::F32, [DType::F32], OpAttrs::Relu { .. })
            | (DType::F64, [DType::F64], OpAttrs::Relu { .. })
            | (DType::F16, [DType::F16], OpAttrs::Relu { .. })
            | (DType::BF16, [DType::BF16], OpAttrs::Relu { .. })
            | (DType::F8E5M2, [DType::F8E5M2], OpAttrs::Relu { .. })
            | (DType::I8, [DType::I8], OpAttrs::Relu { .. })
            | (DType::I16, [DType::I16], OpAttrs::Relu { .. })
            | (DType::I32, [DType::I32], OpAttrs::Relu { .. })
            | (DType::I64, [DType::I64], OpAttrs::Relu { .. })
            | (DType::I4, [DType::I4], OpAttrs::Relu { .. })
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
            TensorValue::BF16(out) => relu_inplace_bf16(attrs, &mut out.data, thread_id),
            TensorValue::F8E5M2(out) => relu_inplace_f8(attrs, &mut out.data, thread_id),
            TensorValue::I8(out) => relu_inplace_i8(attrs, &mut out.data, thread_id),
            TensorValue::I16(out) => relu_inplace_i16(attrs, &mut out.data, thread_id),
            TensorValue::I32(out) => relu_inplace_i32(attrs, &mut out.data, thread_id),
            TensorValue::I64(out) => relu_inplace_i64(attrs, &mut out.data, thread_id),
            TensorValue::I4(out) => {
                let len = out.numel();
                relu_inplace_i4(attrs, &mut out.data, len, thread_id)
            }
            other => Err(anyhow!("relu inplace does not support dtype {:?}", other.dtype())),
        }
    });
    Some(InplaceKernelFn::Host(kernel))
}
