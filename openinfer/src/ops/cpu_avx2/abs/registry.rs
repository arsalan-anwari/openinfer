use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::{cpu_kernel, KernelFn};
use crate::tensor::DType;
use crate::tensor::{I2, I4, Tensor, TensorElement, TensorOptions};

use super::{abs_f32, abs_f64, abs_i16, abs_i32, abs_i64, abs_i8, abs_i4_packed, abs_i2_packed};

pub fn lookup_kernel_cpu_avx2_abs(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_i8 as fn(&[i8] , usize) -> Result<Vec<i8>>,
        ))),
        (DType::I16, [DType::I16], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_i16 as fn(&[i16] , usize) -> Result<Vec<i16>>,
        ))),
        (DType::F32, [DType::F32], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_f32 as fn(&[f32] , usize) -> Result<Vec<f32>>,
        ))),
        (DType::F64, [DType::F64], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_f64 as fn(&[f64] , usize) -> Result<Vec<f64>>,
        ))),
        (DType::I32, [DType::I32], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_i32 as fn(&[i32] , usize) -> Result<Vec<i32>>,
        ))),
        (DType::I64, [DType::I64], &OpAttrs::None) => Some(KernelFn::Host(cpu_kernel(
            abs_i64 as fn(&[i64] , usize) -> Result<Vec<i64>>,
        ))),
        (DType::I4, [DType::I4], &OpAttrs::None) => Some(KernelFn::Host(Box::new(|_, inputs, _output, thread_id| {
            let a = <I4 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("abs input 0 dtype mismatch"))?;
            let out = abs_i4_packed(&a.data, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(Some(<I4 as TensorElement>::into_value(tensor)))
        }))),
        (DType::I2, [DType::I2], &OpAttrs::None) => Some(KernelFn::Host(Box::new(|_, inputs, _output, thread_id| {
            let a = <I2 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("abs input 0 dtype mismatch"))?;
            let out = abs_i2_packed(&a.data, a.numel(), thread_id)?;
            let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                shape: Some(a.shape().to_vec()),
                allow_len_mismatch: true,
                ..TensorOptions::default()
            })?;
            Ok(Some(<I2 as TensorElement>::into_value(tensor)))
        }))),
        _ => None,
    }
}
