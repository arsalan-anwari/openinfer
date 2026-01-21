use anyhow::Result;

use anyhow::anyhow;
use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel, KernelFn};
use crate::tensor::DType;
use crate::tensor::{I1, I2, I4, U1, U2, U4, TensorElement, TensorOptions, Tensor};

use super::{
    add_bf16, add_bitset, add_bool, add_f16, add_f32, add_f64, add_f8, add_i16, add_i32, add_i64, add_i8, add_u16,
    add_u32, add_u64, add_u8, add_i4, add_i2, add_i1, add_u4, add_u2, add_u1,
};

pub fn lookup_kernel_cpu_add(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_i8 as fn(&[i8], &[i8] , usize) -> Result<Vec<i8>>,
        ))),
        (DType::I16, [DType::I16, DType::I16], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_i16 as fn(&[i16], &[i16] , usize) -> Result<Vec<i16>>,
        ))),
        (DType::F32, [DType::F32, DType::F32], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_f32 as fn(&[f32], &[f32] , usize) -> Result<Vec<f32>>,
        ))),
        (DType::F64, [DType::F64, DType::F64], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_f64 as fn(&[f64], &[f64] , usize) -> Result<Vec<f64>>,
        ))),
        (DType::F16, [DType::F16, DType::F16], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_f16 as fn(&[crate::tensor::F16], &[crate::tensor::F16], usize) -> Result<Vec<crate::tensor::F16>>,
        ))),
        (DType::BF16, [DType::BF16, DType::BF16], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_bf16 as fn(&[crate::tensor::BF16], &[crate::tensor::BF16], usize) -> Result<Vec<crate::tensor::BF16>>,
        ))),
        (DType::F8E5M2, [DType::F8E5M2, DType::F8E5M2], &OpAttrs::None) => {
            Some(KernelFn::Host(cpu_kernel(
                add_f8 as fn(&[crate::tensor::F8E5M2], &[crate::tensor::F8E5M2], usize) -> Result<Vec<crate::tensor::F8E5M2>>,
            )))
        }
        (DType::U8, [DType::U8, DType::U8], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_u8 as fn(&[u8], &[u8] , usize) -> Result<Vec<u8>>,
        ))),
        (DType::U16, [DType::U16, DType::U16], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_u16 as fn(&[u16], &[u16] , usize) -> Result<Vec<u16>>,
        ))),
        (DType::I32, [DType::I32, DType::I32], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_i32 as fn(&[i32], &[i32] , usize) -> Result<Vec<i32>>,
        ))),
        (DType::I64, [DType::I64, DType::I64], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_i64 as fn(&[i64], &[i64] , usize) -> Result<Vec<i64>>,
        ))),
        (DType::U32, [DType::U32, DType::U32], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_u32 as fn(&[u32], &[u32] , usize) -> Result<Vec<u32>>,
        ))),
        (DType::U64, [DType::U64, DType::U64], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_u64 as fn(&[u64], &[u64] , usize) -> Result<Vec<u64>>,
        ))),
        (DType::Bool, [DType::Bool, DType::Bool], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            add_bool as fn(&[bool], &[bool] , usize) -> Result<Vec<bool>>,
        ))),
        (DType::Bitset, [DType::Bitset, DType::Bitset], &OpAttrs::None) => {
            Some(KernelFn::Host(cpu_kernel(
                add_bitset as fn(&[crate::tensor::Bitset], &[crate::tensor::Bitset], usize) -> Result<Vec<crate::tensor::Bitset>>,
            )))
        }
        (DType::I4, [DType::I4, DType::I4], &OpAttrs::None) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                let a = <I4 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <I4 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i4(&a.data, &b.data, a.numel(), thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(a.shape().to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })?;
                Ok(<I4 as TensorElement>::into_value(tensor))
            })))
        }
        (DType::I2, [DType::I2, DType::I2], &OpAttrs::None) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                let a = <I2 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <I2 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i2(&a.data, &b.data, a.numel(), thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(a.shape().to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })?;
                Ok(<I2 as TensorElement>::into_value(tensor))
            })))
        }
        (DType::I1, [DType::I1, DType::I1], &OpAttrs::None) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                let a = <I1 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <I1 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let out = add_i1(&a.data, &b.data, a.numel(), thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(a.shape().to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })?;
                Ok(<I1 as TensorElement>::into_value(tensor))
            })))
        }
        (DType::U4, [DType::U4, DType::U4], &OpAttrs::None) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                let a = <U4 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <U4 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u4(&a.data, &b.data, a.numel(), thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(a.shape().to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })?;
                Ok(<U4 as TensorElement>::into_value(tensor))
            })))
        }
        (DType::U2, [DType::U2, DType::U2], &OpAttrs::None) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                let a = <U2 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <U2 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u2(&a.data, &b.data, a.numel(), thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(a.shape().to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })?;
                Ok(<U2 as TensorElement>::into_value(tensor))
            })))
        }
        (DType::U1, [DType::U1, DType::U1], &OpAttrs::None) => {
            Some(KernelFn::Host(Box::new(|_, inputs, thread_id| {
                let a = <U1 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <U1 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let out = add_u1(&a.data, &b.data, a.numel(), thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(a.shape().to_vec()),
                    allow_len_mismatch: true,
                    ..TensorOptions::default()
                })?;
                Ok(<U1 as TensorElement>::into_value(tensor))
            })))
        }
        _ => None,
    }
}
