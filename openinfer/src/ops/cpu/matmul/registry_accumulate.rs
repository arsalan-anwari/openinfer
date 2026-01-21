use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::registry::KernelFn;
use crate::tensor::{DType, Tensor, TensorElement, TensorValue};

use super::{
    matmul_i16_i32, matmul_i16_i64, matmul_i32_i64, matmul_i8_i16, matmul_i8_i32, matmul_i8_i64,
    matmul_u16_u32, matmul_u16_u64, matmul_u32_u64, matmul_u8_u16, matmul_u8_u32, matmul_u8_u64,
};

fn matmul_acc_kernel<I, O>(
    inputs: &[TensorValue],
    thread_id: usize,
    func: fn(&Tensor<I>, &Tensor<I>, usize) -> anyhow::Result<Tensor<O>>,
) -> anyhow::Result<TensorValue>
where
    I: TensorElement + Clone + 'static,
    O: TensorElement + Clone + 'static,
{
    if inputs.len() < 2 {
        return Err(anyhow!("matmul op expects 2 inputs"));
    }
    let a = I::from_value(&inputs[0]).ok_or_else(|| anyhow!("matmul input 0 dtype mismatch"))?;
    let b = I::from_value(&inputs[1]).ok_or_else(|| anyhow!("matmul input 1 dtype mismatch"))?;
    let out = func(&a, &b, thread_id)?;
    Ok(O::into_value(out))
}

pub fn lookup_kernel_cpu_matmul_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I16, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_acc_kernel(inputs, thread_id, matmul_i8_i16)
            })))
        }
        (DType::I32, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_acc_kernel(inputs, thread_id, matmul_i8_i32)
            })))
        }
        (DType::I64, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_acc_kernel(inputs, thread_id, matmul_i8_i64)
            })))
        }
        (DType::I32, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_acc_kernel(inputs, thread_id, matmul_i16_i32)
            })))
        }
        (DType::I64, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_acc_kernel(inputs, thread_id, matmul_i16_i64)
            })))
        }
        (DType::I64, [DType::I32, DType::I32], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_acc_kernel(inputs, thread_id, matmul_i32_i64)
            })))
        }
        (DType::U16, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_acc_kernel(inputs, thread_id, matmul_u8_u16)
            })))
        }
        (DType::U32, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_acc_kernel(inputs, thread_id, matmul_u8_u32)
            })))
        }
        (DType::U64, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_acc_kernel(inputs, thread_id, matmul_u8_u64)
            })))
        }
        (DType::U32, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_acc_kernel(inputs, thread_id, matmul_u16_u32)
            })))
        }
        (DType::U64, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_acc_kernel(inputs, thread_id, matmul_u16_u64)
            })))
        }
        (DType::U64, [DType::U32, DType::U32], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                matmul_acc_kernel(inputs, thread_id, matmul_u32_u64)
            })))
        }
        _ => None,
    }
}
