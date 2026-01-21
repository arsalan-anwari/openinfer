use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel_out, KernelFn};
use crate::tensor::{DType, I1, I2, I4, U1, U2, U4, TensorElement};

use super::{
    add_i16_i32, add_i16_i64, add_i32_i64, add_i8_i16, add_i8_i32, add_i8_i64, add_u16_u32,
    add_u16_u64, add_u32_u64, add_u8_u16, add_u8_u32, add_u8_u64, add_i4_i8_packed,
    add_i4_i16_packed, add_i4_i32_packed, add_i4_i64_packed, add_i2_i8_packed, add_i2_i16_packed,
    add_i2_i32_packed, add_i2_i64_packed, add_i1_i8_packed, add_i1_i16_packed, add_i1_i32_packed,
    add_i1_i64_packed, add_u4_u8_packed, add_u4_u16_packed, add_u4_u32_packed, add_u4_u64_packed,
    add_u2_u8_packed, add_u2_u16_packed, add_u2_u32_packed, add_u2_u64_packed, add_u1_u8_packed,
    add_u1_u16_packed, add_u1_u32_packed, add_u1_u64_packed,
};

pub fn lookup_kernel_cpu_add_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I16, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_i8_i16 as fn(&[i8], &[i8], usize) -> anyhow::Result<Vec<i16>>,
            )))
        }
        (DType::I32, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_i8_i32 as fn(&[i8], &[i8], usize) -> anyhow::Result<Vec<i32>>,
            )))
        }
        (DType::I64, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_i8_i64 as fn(&[i8], &[i8], usize) -> anyhow::Result<Vec<i64>>,
            )))
        }
        (DType::I32, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_i16_i32 as fn(&[i16], &[i16], usize) -> anyhow::Result<Vec<i32>>,
            )))
        }
        (DType::I64, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_i16_i64 as fn(&[i16], &[i16], usize) -> anyhow::Result<Vec<i64>>,
            )))
        }
        (DType::I64, [DType::I32, DType::I32], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_i32_i64 as fn(&[i32], &[i32], usize) -> anyhow::Result<Vec<i64>>,
            )))
        }
        (DType::U16, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_u8_u16 as fn(&[u8], &[u8], usize) -> anyhow::Result<Vec<u16>>,
            )))
        }
        (DType::U32, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_u8_u32 as fn(&[u8], &[u8], usize) -> anyhow::Result<Vec<u32>>,
            )))
        }
        (DType::U64, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_u8_u64 as fn(&[u8], &[u8], usize) -> anyhow::Result<Vec<u64>>,
            )))
        }
        (DType::U32, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_u16_u32 as fn(&[u16], &[u16], usize) -> anyhow::Result<Vec<u32>>,
            )))
        }
        (DType::U64, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_u16_u64 as fn(&[u16], &[u16], usize) -> anyhow::Result<Vec<u64>>,
            )))
        }
        (DType::U64, [DType::U32, DType::U32], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(KernelFn::Host(cpu_kernel_out(
                add_u32_u64 as fn(&[u32], &[u32], usize) -> anyhow::Result<Vec<u64>>,
            )))
        }
        (DType::I8, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <I4 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i4_i8_packed(&a, &b, thread_id)?;
                Ok(<i8 as TensorElement>::into_value(out))
            })))
        }
        (DType::I16, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <I4 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i4_i16_packed(&a, &b, thread_id)?;
                Ok(<i16 as TensorElement>::into_value(out))
            })))
        }
        (DType::I32, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <I4 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i4_i32_packed(&a, &b, thread_id)?;
                Ok(<i32 as TensorElement>::into_value(out))
            })))
        }
        (DType::I64, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <I4 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i4_i64_packed(&a, &b, thread_id)?;
                Ok(<i64 as TensorElement>::into_value(out))
            })))
        }
        (DType::I8, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <I2 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i2_i8_packed(&a, &b, thread_id)?;
                Ok(<i8 as TensorElement>::into_value(out))
            })))
        }
        (DType::I16, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <I2 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i2_i16_packed(&a, &b, thread_id)?;
                Ok(<i16 as TensorElement>::into_value(out))
            })))
        }
        (DType::I32, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <I2 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i2_i32_packed(&a, &b, thread_id)?;
                Ok(<i32 as TensorElement>::into_value(out))
            })))
        }
        (DType::I64, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <I2 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i2_i64_packed(&a, &b, thread_id)?;
                Ok(<i64 as TensorElement>::into_value(out))
            })))
        }
        (DType::I8, [DType::I1, DType::I1], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I1 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <I1 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i1_i8_packed(&a, &b, thread_id)?;
                Ok(<i8 as TensorElement>::into_value(out))
            })))
        }
        (DType::I16, [DType::I1, DType::I1], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I1 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <I1 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i1_i16_packed(&a, &b, thread_id)?;
                Ok(<i16 as TensorElement>::into_value(out))
            })))
        }
        (DType::I32, [DType::I1, DType::I1], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I1 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <I1 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i1_i32_packed(&a, &b, thread_id)?;
                Ok(<i32 as TensorElement>::into_value(out))
            })))
        }
        (DType::I64, [DType::I1, DType::I1], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <I1 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <I1 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i1_i64_packed(&a, &b, thread_id)?;
                Ok(<i64 as TensorElement>::into_value(out))
            })))
        }
        (DType::U8, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U8 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <U4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <U4 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u4_u8_packed(&a, &b, thread_id)?;
                Ok(<u8 as TensorElement>::into_value(out))
            })))
        }
        (DType::U16, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <U4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <U4 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u4_u16_packed(&a, &b, thread_id)?;
                Ok(<u16 as TensorElement>::into_value(out))
            })))
        }
        (DType::U32, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <U4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <U4 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u4_u32_packed(&a, &b, thread_id)?;
                Ok(<u32 as TensorElement>::into_value(out))
            })))
        }
        (DType::U64, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <U4 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <U4 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u4_u64_packed(&a, &b, thread_id)?;
                Ok(<u64 as TensorElement>::into_value(out))
            })))
        }
        (DType::U8, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U8 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <U2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <U2 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u2_u8_packed(&a, &b, thread_id)?;
                Ok(<u8 as TensorElement>::into_value(out))
            })))
        }
        (DType::U16, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <U2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <U2 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u2_u16_packed(&a, &b, thread_id)?;
                Ok(<u16 as TensorElement>::into_value(out))
            })))
        }
        (DType::U32, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <U2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <U2 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u2_u32_packed(&a, &b, thread_id)?;
                Ok(<u32 as TensorElement>::into_value(out))
            })))
        }
        (DType::U64, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <U2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <U2 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u2_u64_packed(&a, &b, thread_id)?;
                Ok(<u64 as TensorElement>::into_value(out))
            })))
        }
        (DType::U8, [DType::U1, DType::U1], &OpAttrs::Accumulate { dtype: DType::U8 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <U1 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <U1 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u1_u8_packed(&a, &b, thread_id)?;
                Ok(<u8 as TensorElement>::into_value(out))
            })))
        }
        (DType::U16, [DType::U1, DType::U1], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <U1 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <U1 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u1_u16_packed(&a, &b, thread_id)?;
                Ok(<u16 as TensorElement>::into_value(out))
            })))
        }
        (DType::U32, [DType::U1, DType::U1], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <U1 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <U1 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u1_u32_packed(&a, &b, thread_id)?;
                Ok(<u32 as TensorElement>::into_value(out))
            })))
        }
        (DType::U64, [DType::U1, DType::U1], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(KernelFn::Host(Box::new(|_attrs, inputs, thread_id| {
                let a = <U1 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("add input 0 dtype mismatch"))?;
                let b = <U1 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u1_u64_packed(&a, &b, thread_id)?;
                Ok(<u64 as TensorElement>::into_value(out))
            })))
        }
        _ => None,
    }
}
