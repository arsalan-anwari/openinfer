use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel, KernelFn};
use crate::tensor::DType;
use crate::tensor::{I2, I4, U2, U4, Tensor, TensorElement, TensorOptions};

use super::{
    mul_bool, mul_f32, mul_f64, mul_i16, mul_i32, mul_i64, mul_i8, mul_u16, mul_u32, mul_u64,
    mul_u8, mul_i4_packed, mul_i2_packed, mul_u4_packed, mul_u2_packed,
};

pub fn lookup_kernel_cpu_avx_mul(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            mul_i8 as fn(&[i8], &[i8] , usize) -> Result<Vec<i8>>,
        ))),
        (DType::I16, [DType::I16, DType::I16], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            mul_i16 as fn(&[i16], &[i16] , usize) -> Result<Vec<i16>>,
        ))),
        (DType::F32, [DType::F32, DType::F32], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            mul_f32 as fn(&[f32], &[f32] , usize) -> Result<Vec<f32>>,
        ))),
        (DType::F64, [DType::F64, DType::F64], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            mul_f64 as fn(&[f64], &[f64] , usize) -> Result<Vec<f64>>,
        ))),
        (DType::U8, [DType::U8, DType::U8], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            mul_u8 as fn(&[u8], &[u8] , usize) -> Result<Vec<u8>>,
        ))),
        (DType::U16, [DType::U16, DType::U16], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            mul_u16 as fn(&[u16], &[u16] , usize) -> Result<Vec<u16>>,
        ))),
        (DType::I32, [DType::I32, DType::I32], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            mul_i32 as fn(&[i32], &[i32] , usize) -> Result<Vec<i32>>,
        ))),
        (DType::I64, [DType::I64, DType::I64], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            mul_i64 as fn(&[i64], &[i64] , usize) -> Result<Vec<i64>>,
        ))),
        (DType::U32, [DType::U32, DType::U32], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            mul_u32 as fn(&[u32], &[u32] , usize) -> Result<Vec<u32>>,
        ))),
        (DType::U64, [DType::U64, DType::U64], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            mul_u64 as fn(&[u64], &[u64] , usize) -> Result<Vec<u64>>,
        ))),
        (DType::Bool, [DType::Bool, DType::Bool], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            mul_bool as fn(&[bool], &[bool] , usize) -> Result<Vec<bool>>,
        ))),
        (DType::I4, [DType::I4, DType::I4], &OpAttrs::None) => Some(KernelFn::Host(Box::new(|_, inputs, _output, thread_id| {
            let a = <I4 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("mul input 0 dtype mismatch"))?;
            let b = <I4 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("mul input 1 dtype mismatch"))?;
            let out = mul_i4_packed(&a.data, &b.data, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(Some(<I4 as TensorElement>::into_value(tensor)))
        }))),
        (DType::I2, [DType::I2, DType::I2], &OpAttrs::None) => Some(KernelFn::Host(Box::new(|_, inputs, _output, thread_id| {
            let a = <I2 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("mul input 0 dtype mismatch"))?;
            let b = <I2 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("mul input 1 dtype mismatch"))?;
            let out = mul_i2_packed(&a.data, &b.data, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(Some(<I2 as TensorElement>::into_value(tensor)))
        }))),
        (DType::U4, [DType::U4, DType::U4], &OpAttrs::None) => Some(KernelFn::Host(Box::new(|_, inputs, _output, thread_id| {
            let a = <U4 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("mul input 0 dtype mismatch"))?;
            let b = <U4 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("mul input 1 dtype mismatch"))?;
            let out = mul_u4_packed(&a.data, &b.data, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(Some(<U4 as TensorElement>::into_value(tensor)))
        }))),
        (DType::U2, [DType::U2, DType::U2], &OpAttrs::None) => Some(KernelFn::Host(Box::new(|_, inputs, _output, thread_id| {
            let a = <U2 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("mul input 0 dtype mismatch"))?;
            let b = <U2 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("mul input 1 dtype mismatch"))?;
            let out = mul_u2_packed(&a.data, &b.data, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(Some(<U2 as TensorElement>::into_value(tensor)))
        }))),
        _ => None,
    }
}
