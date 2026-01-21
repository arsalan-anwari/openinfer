use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel_out, KernelFn};
use crate::tensor::{DType, I1, I2, I4, TensorElement};

use super::{
    abs_i16_i32, abs_i16_i64, abs_i32_i64, abs_i8_i16, abs_i8_i32, abs_i8_i64, abs_i4_i8_packed,
    abs_i4_i16_packed, abs_i4_i32_packed, abs_i4_i64_packed, abs_i2_i8_packed, abs_i2_i16_packed,
    abs_i2_i32_packed, abs_i2_i64_packed, abs_i1_i8_packed, abs_i1_i16_packed, abs_i1_i32_packed,
    abs_i1_i64_packed,
};

pub fn lookup_kernel_cpu_abs_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I16, [DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                abs_i8_i16 as fn(&[i8], usize) -> anyhow::Result<Vec<i16>>,
            )))
        }
        (DType::I32, [DType::I8], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                abs_i8_i32 as fn(&[i8], usize) -> anyhow::Result<Vec<i32>>,
            )))
        }
        (DType::I64, [DType::I8], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                abs_i8_i64 as fn(&[i8], usize) -> anyhow::Result<Vec<i64>>,
            )))
        }
        (DType::I32, [DType::I16], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                abs_i16_i32 as fn(&[i16], usize) -> anyhow::Result<Vec<i32>>,
            )))
        }
        (DType::I64, [DType::I16], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                abs_i16_i64 as fn(&[i16], usize) -> anyhow::Result<Vec<i64>>,
            )))
        }
        (DType::I64, [DType::I32], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                abs_i32_i64 as fn(&[i32], usize) -> anyhow::Result<Vec<i64>>,
            )))
        }
        (DType::I8, [DType::I4], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out = abs_i4_i8_packed(&a, thread_id)?;
                Ok(<i8 as TensorElement>::into_value(out))
            })))
        }
        (DType::I16, [DType::I4], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out = abs_i4_i16_packed(&a, thread_id)?;
                Ok(<i16 as TensorElement>::into_value(out))
            })))
        }
        (DType::I32, [DType::I4], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out = abs_i4_i32_packed(&a, thread_id)?;
                Ok(<i32 as TensorElement>::into_value(out))
            })))
        }
        (DType::I64, [DType::I4], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out = abs_i4_i64_packed(&a, thread_id)?;
                Ok(<i64 as TensorElement>::into_value(out))
            })))
        }
        (DType::I8, [DType::I2], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out = abs_i2_i8_packed(&a, thread_id)?;
                Ok(<i8 as TensorElement>::into_value(out))
            })))
        }
        (DType::I16, [DType::I2], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out = abs_i2_i16_packed(&a, thread_id)?;
                Ok(<i16 as TensorElement>::into_value(out))
            })))
        }
        (DType::I32, [DType::I2], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out = abs_i2_i32_packed(&a, thread_id)?;
                Ok(<i32 as TensorElement>::into_value(out))
            })))
        }
        (DType::I64, [DType::I2], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out = abs_i2_i64_packed(&a, thread_id)?;
                Ok(<i64 as TensorElement>::into_value(out))
            })))
        }
        (DType::I8, [DType::I1], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I1 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out = abs_i1_i8_packed(&a, thread_id)?;
                Ok(<i8 as TensorElement>::into_value(out))
            })))
        }
        (DType::I16, [DType::I1], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I1 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out = abs_i1_i16_packed(&a, thread_id)?;
                Ok(<i16 as TensorElement>::into_value(out))
            })))
        }
        (DType::I32, [DType::I1], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I1 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out = abs_i1_i32_packed(&a, thread_id)?;
                Ok(<i32 as TensorElement>::into_value(out))
            })))
        }
        (DType::I64, [DType::I1], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I1 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out = abs_i1_i64_packed(&a, thread_id)?;
                Ok(<i64 as TensorElement>::into_value(out))
            })))
        }
        _ => None,
    }
}
