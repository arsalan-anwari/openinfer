use crate::graph::OpAttrs;
use crate::ops::registry::KernelFn;
use crate::tensor::{DType, TensorElement};

use super::{is_finite_bf16, is_finite_f16, is_finite_f32, is_finite_f64, is_finite_f8};

pub fn lookup_kernel_cpu_is_finite(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::Bool, [DType::F8E5M2], &OpAttrs::None) => Some(KernelFn::Host(Box::new(|_, inputs, _output, thread_id| {
            let tensor = crate::tensor::F8E5M2::from_value(&inputs[0])
                .ok_or_else(|| anyhow::anyhow!("is_finite input 0 dtype mismatch"))?;
            Ok(Some(is_finite_f8(&tensor, thread_id)?))
        }))),
        (DType::Bool, [DType::BF16], &OpAttrs::None) => Some(KernelFn::Host(Box::new(|_, inputs, _output, thread_id| {
            let tensor = crate::tensor::BF16::from_value(&inputs[0])
                .ok_or_else(|| anyhow::anyhow!("is_finite input 0 dtype mismatch"))?;
            Ok(Some(is_finite_bf16(&tensor, thread_id)?))
        }))),
        (DType::Bool, [DType::F16], &OpAttrs::None) => Some(KernelFn::Host(Box::new(|_, inputs, _output, thread_id| {
            let tensor = crate::tensor::F16::from_value(&inputs[0])
                .ok_or_else(|| anyhow::anyhow!("is_finite input 0 dtype mismatch"))?;
            Ok(Some(is_finite_f16(&tensor, thread_id)?))
        }))),
        (DType::Bool, [DType::F32], &OpAttrs::None) => Some(KernelFn::Host(Box::new(|_, inputs, _output, thread_id| {
            let tensor = f32::from_value(&inputs[0])
                .ok_or_else(|| anyhow::anyhow!("is_finite input 0 dtype mismatch"))?;
            Ok(Some(is_finite_f32(&tensor, thread_id)?))
        }))),
        (DType::Bool, [DType::F64], &OpAttrs::None) => Some(KernelFn::Host(Box::new(|_, inputs, _output, thread_id| {
            let tensor = f64::from_value(&inputs[0])
                .ok_or_else(|| anyhow::anyhow!("is_finite input 0 dtype mismatch"))?;
            Ok(Some(is_finite_f64(&tensor, thread_id)?))
        }))),
        _ => None,
    }
}
