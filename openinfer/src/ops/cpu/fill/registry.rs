use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel, KernelFn};
use crate::tensor::{DType, I1, I2, I4, U1, U2, U4, Tensor, TensorElement, TensorOptions};

use super::{
    fill_bf16, fill_bitset, fill_bool, fill_f16, fill_f32, fill_f64, fill_f8, fill_i16, fill_i32,
    fill_i64, fill_i8, fill_u16, fill_u32, fill_u64, fill_u8, fill_i4, fill_i2, fill_i1,
    fill_u4, fill_u2, fill_u1,
};

pub fn lookup_kernel_cpu_fill(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_i8 as fn(&OpAttrs, &[i8], usize) -> Result<Vec<i8>>),
        )),
        (DType::I16, [DType::I16], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_i16 as fn(&OpAttrs, &[i16], usize) -> Result<Vec<i16>>),
        )),
        (DType::I32, [DType::I32], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_i32 as fn(&OpAttrs, &[i32], usize) -> Result<Vec<i32>>),
        )),
        (DType::I64, [DType::I64], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_i64 as fn(&OpAttrs, &[i64], usize) -> Result<Vec<i64>>),
        )),
        (DType::U8, [DType::U8], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_u8 as fn(&OpAttrs, &[u8], usize) -> Result<Vec<u8>>),
        )),
        (DType::U16, [DType::U16], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_u16 as fn(&OpAttrs, &[u16], usize) -> Result<Vec<u16>>),
        )),
        (DType::U32, [DType::U32], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_u32 as fn(&OpAttrs, &[u32], usize) -> Result<Vec<u32>>),
        )),
        (DType::U64, [DType::U64], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_u64 as fn(&OpAttrs, &[u64], usize) -> Result<Vec<u64>>),
        )),
        (DType::F16, [DType::F16], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_f16 as fn(&OpAttrs, &[crate::tensor::F16], usize) -> Result<Vec<crate::tensor::F16>>),
        )),
        (DType::BF16, [DType::BF16], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_bf16 as fn(&OpAttrs, &[crate::tensor::BF16], usize) -> Result<Vec<crate::tensor::BF16>>),
        )),
        (DType::F8E5M2, [DType::F8E5M2], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_f8 as fn(&OpAttrs, &[crate::tensor::F8E5M2], usize) -> Result<Vec<crate::tensor::F8E5M2>>),
        )),
        (DType::F32, [DType::F32], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_f32 as fn(&OpAttrs, &[f32], usize) -> Result<Vec<f32>>),
        )),
        (DType::F64, [DType::F64], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_f64 as fn(&OpAttrs, &[f64], usize) -> Result<Vec<f64>>),
        )),
        (DType::Bool, [DType::Bool], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(fill_bool as fn(&OpAttrs, &[bool], usize) -> Result<Vec<bool>>),
        )),
        (DType::Bitset, [DType::Bitset], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(
            cpu_kernel(
                fill_bitset
                    as fn(
                        &OpAttrs,
                        &[crate::tensor::Bitset],
                        usize,
                    ) -> Result<Vec<crate::tensor::Bitset>>,
            ),
        )),
        (DType::I4, [DType::I4], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(Box::new(|attrs, inputs, thread_id| {
            let a = <I4 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("fill input 0 dtype mismatch"))?;
            let out = fill_i4(attrs, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(<I4 as TensorElement>::into_value(tensor))
        }))),
        (DType::I2, [DType::I2], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(Box::new(|attrs, inputs, thread_id| {
            let a = <I2 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("fill input 0 dtype mismatch"))?;
            let out = fill_i2(attrs, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(<I2 as TensorElement>::into_value(tensor))
        }))),
        (DType::I1, [DType::I1], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(Box::new(|attrs, inputs, thread_id| {
            let a = <I1 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("fill input 0 dtype mismatch"))?;
            let out = fill_i1(attrs, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(<I1 as TensorElement>::into_value(tensor))
        }))),
        (DType::U4, [DType::U4], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(Box::new(|attrs, inputs, thread_id| {
            let a = <U4 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("fill input 0 dtype mismatch"))?;
            let out = fill_u4(attrs, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(<U4 as TensorElement>::into_value(tensor))
        }))),
        (DType::U2, [DType::U2], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(Box::new(|attrs, inputs, thread_id| {
            let a = <U2 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("fill input 0 dtype mismatch"))?;
            let out = fill_u2(attrs, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(<U2 as TensorElement>::into_value(tensor))
        }))),
        (DType::U1, [DType::U1], &OpAttrs::Fill { .. }) => Some(KernelFn::Host(Box::new(|attrs, inputs, thread_id| {
            let a = <U1 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("fill input 0 dtype mismatch"))?;
            let out = fill_u1(attrs, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(<U1 as TensorElement>::into_value(tensor))
        }))),
        _ => None,
    }
}
