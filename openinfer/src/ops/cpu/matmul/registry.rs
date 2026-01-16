use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::registry::KernelFn;
use crate::tensor::{Tensor, TensorElement, TensorValue, DType};

use super::{
    matmul_bitset, matmul_bool, matmul_f16, matmul_f32, matmul_f64, matmul_i16, matmul_i32,
    matmul_i64, matmul_i8, matmul_u16, matmul_u32, matmul_u64, matmul_u8,
};

fn matmul_kernel<T>(
    inputs: &[TensorValue],
    thread_id: usize,
    func: fn(&Tensor<T>, &Tensor<T>, usize) -> Result<Tensor<T>>,
) -> Result<TensorValue>
where
    T: TensorElement + Clone + 'static,
{
    if inputs.len() < 2 {
        return Err(anyhow!("matmul op expects 2 inputs"));
    }
    let a = T::from_value(&inputs[0]).ok_or_else(|| anyhow!("matmul input 0 dtype mismatch"))?;
    let b = T::from_value(&inputs[1]).ok_or_else(|| anyhow!("matmul input 1 dtype mismatch"))?;
    let out = func(&a, &b, thread_id)?;
    Ok(T::into_value(out))
}

pub fn lookup_kernel_cpu_matmul(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], &OpAttrs::None) => Some(KernelFn::Host(Box::new(
            |_, inputs, thread_id| matmul_kernel(inputs, thread_id, matmul_i8),
        ))),
        (DType::I16, [DType::I16, DType::I16], &OpAttrs::None) => Some(KernelFn::Host(Box::new(
            |_, inputs, thread_id| matmul_kernel(inputs, thread_id, matmul_i16),
        ))),
        (DType::I32, [DType::I32, DType::I32], &OpAttrs::None) => Some(KernelFn::Host(Box::new(
            |_, inputs, thread_id| matmul_kernel(inputs, thread_id, matmul_i32),
        ))),
        (DType::I64, [DType::I64, DType::I64], &OpAttrs::None) => Some(KernelFn::Host(Box::new(
            |_, inputs, thread_id| matmul_kernel(inputs, thread_id, matmul_i64),
        ))),
        (DType::U8, [DType::U8, DType::U8], &OpAttrs::None) => Some(KernelFn::Host(Box::new(
            |_, inputs, thread_id| matmul_kernel(inputs, thread_id, matmul_u8),
        ))),
        (DType::U16, [DType::U16, DType::U16], &OpAttrs::None) => Some(KernelFn::Host(Box::new(
            |_, inputs, thread_id| matmul_kernel(inputs, thread_id, matmul_u16),
        ))),
        (DType::U32, [DType::U32, DType::U32], &OpAttrs::None) => Some(KernelFn::Host(Box::new(
            |_, inputs, thread_id| matmul_kernel(inputs, thread_id, matmul_u32),
        ))),
        (DType::U64, [DType::U64, DType::U64], &OpAttrs::None) => Some(KernelFn::Host(Box::new(
            |_, inputs, thread_id| matmul_kernel(inputs, thread_id, matmul_u64),
        ))),
        (DType::F16, [DType::F16, DType::F16], &OpAttrs::None) => Some(KernelFn::Host(Box::new(
            |_, inputs, thread_id| matmul_kernel(inputs, thread_id, matmul_f16),
        ))),
        (DType::F32, [DType::F32, DType::F32], &OpAttrs::None) => Some(KernelFn::Host(Box::new(
            |_, inputs, thread_id| matmul_kernel(inputs, thread_id, matmul_f32),
        ))),
        (DType::F64, [DType::F64, DType::F64], &OpAttrs::None) => Some(KernelFn::Host(Box::new(
            |_, inputs, thread_id| matmul_kernel(inputs, thread_id, matmul_f64),
        ))),
        (DType::Bool, [DType::Bool, DType::Bool], &OpAttrs::None) => Some(KernelFn::Host(Box::new(
            |_, inputs, thread_id| matmul_kernel(inputs, thread_id, matmul_bool),
        ))),
        (DType::Bitset, [DType::Bitset, DType::Bitset], &OpAttrs::None) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_kernel(inputs, thread_id, matmul_bitset)
            })))
        }
        _ => None,
    }
}
