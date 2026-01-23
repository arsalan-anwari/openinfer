use anyhow::anyhow;
use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::DType;
use crate::tensor::{I1, I2, I4, U1, U2, U4, TensorElement, TensorOptions, Tensor};

use super::{
    add_tensor_bf16, add_tensor_bitset, add_tensor_bool, add_tensor_f16, add_tensor_f32,
    add_tensor_f64, add_tensor_f8, add_tensor_i16, add_tensor_i32, add_tensor_i64, add_tensor_i8,
    add_tensor_u16, add_tensor_u32, add_tensor_u64, add_tensor_u8, add_i4, add_i2, add_i1, add_u4,
    add_u2, add_u1,
};

pub fn lookup_kernel_cpu_add(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I8, [DType::I8, DType::I8], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <i8 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <i8 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_i8(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<i8 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::I16, [DType::I16, DType::I16], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <i16 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <i16 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_i16(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<i16 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::F32, [DType::F32, DType::F32], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <f32 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <f32 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_f32(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<f32 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::F64, [DType::F64, DType::F64], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <f64 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <f64 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_f64(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<f64 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::F16, [DType::F16, DType::F16], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <crate::tensor::F16 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <crate::tensor::F16 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_f16(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<crate::tensor::F16 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::BF16, [DType::BF16, DType::BF16], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <crate::tensor::BF16 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <crate::tensor::BF16 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_bf16(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<crate::tensor::BF16 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::F8E5M2, [DType::F8E5M2, DType::F8E5M2], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <crate::tensor::F8E5M2 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <crate::tensor::F8E5M2 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_f8(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<crate::tensor::F8E5M2 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::U8, [DType::U8, DType::U8], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <u8 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <u8 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_u8(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<u8 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::U16, [DType::U16, DType::U16], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <u16 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <u16 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_u16(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<u16 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::I32, [DType::I32, DType::I32], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <i32 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <i32 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_i32(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<i32 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::I64, [DType::I64, DType::I64], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <i64 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <i64 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_i64(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<i64 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::U32, [DType::U32, DType::U32], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <u32 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <u32 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_u32(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<u32 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::U64, [DType::U64, DType::U64], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <u64 as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <u64 as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_u64(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<u64 as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::Bool, [DType::Bool, DType::Bool], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <bool as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <bool as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_bool(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<bool as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::Bitset, [DType::Bitset, DType::Bitset], &OpAttrs::None) => Some(KernelFn::Host(
            crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <crate::tensor::Bitset as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <crate::tensor::Bitset as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                let (out, shape) = add_tensor_bitset(&a, &b, thread_id)?;
                let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                    shape: Some(shape),
                    ..TensorOptions::default()
                })?;
                Ok(<crate::tensor::Bitset as TensorElement>::into_value(tensor))
            }),
        )),
        (DType::I4, [DType::I4, DType::I4], &OpAttrs::None) => {
            Some(KernelFn::Host(crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <I4 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <I4 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                if a.shape() != b.shape() {
                    return Err(anyhow!("add packed i4 does not support broadcast"));
                }
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
            Some(KernelFn::Host(crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <I2 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <I2 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                if a.shape() != b.shape() {
                    return Err(anyhow!("add packed i2 does not support broadcast"));
                }
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
            Some(KernelFn::Host(crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <I1 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <I1 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                if a.shape() != b.shape() {
                    return Err(anyhow!("add packed i1 does not support broadcast"));
                }
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
            Some(KernelFn::Host(crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <U4 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <U4 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                if a.shape() != b.shape() {
                    return Err(anyhow!("add packed u4 does not support broadcast"));
                }
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
            Some(KernelFn::Host(crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <U2 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <U2 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                if a.shape() != b.shape() {
                    return Err(anyhow!("add packed u2 does not support broadcast"));
                }
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
            Some(KernelFn::Host(crate::ops::adapter::host_kernel_simple(|_, inputs, thread_id| {
                let a = <U1 as TensorElement>::from_value(&inputs[0]).ok_or_else(|| anyhow!("add input 0 dtype mismatch"))?;
                let b = <U1 as TensorElement>::from_value(&inputs[1]).ok_or_else(|| anyhow!("add input 1 dtype mismatch"))?;
                if a.shape() != b.shape() {
                    return Err(anyhow!("add packed u1 does not support broadcast"));
                }
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
