use crate::graph::OpAttrs;
use crate::ops::registry::KernelFn;
use crate::tensor::{DType, Tensor, TensorElement, TensorOptions, TensorValue};

use super::{
    matmul_i16_i32, matmul_i16_i64, matmul_i32_i64, matmul_i8_i16, matmul_i8_i32, matmul_i8_i64,
    matmul_u16_u32, matmul_u16_u64, matmul_u32_u64, matmul_u8_u16, matmul_u8_u32, matmul_u8_u64,
    matmul_dims,
};

macro_rules! acc_matmul_kernel {
    ($in:ty, $out_ty:ty, $out_variant:ident, $func:ident) => {
        KernelFn::Host(Box::new(|_attrs, inputs, output, thread_id| {
            if inputs.len() < 2 {
                return Err(anyhow::anyhow!("matmul op expects 2 inputs"));
            }
            let a = <$in as TensorElement>::from_value(&inputs[0])
                .ok_or_else(|| anyhow::anyhow!("matmul input 0 dtype mismatch"))?;
            let b = <$in as TensorElement>::from_value(&inputs[1])
                .ok_or_else(|| anyhow::anyhow!("matmul input 1 dtype mismatch"))?;
            let out_slice = output.and_then(|out| match out {
                TensorValue::$out_variant(t) => Some(&mut t.data),
                _ => None,
            });
            let out = $func(
                &a,
                &b,
                out_slice.map(|out| out.as_mut_slice()),
                thread_id,
            )?;
            match out {
                Some(out) => {
                    let (_, m, _, n) = matmul_dims(a.shape(), b.shape())?;
                    let shape: Vec<usize> = a.shape()[..a.shape().len() - 2]
                        .iter()
                        .cloned()
                        .chain([m, n])
                        .collect();
                    let tensor = Tensor::from_vec_with_opts(out, TensorOptions {
                        shape: Some(shape),
                        ..TensorOptions::default()
                    })?;
                    Ok(Some(<$out_ty as TensorElement>::into_value(tensor)))
                }
                None => Ok(None),
            }
        }))
    };
}

pub fn lookup_kernel_cpu_matmul_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I16, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(acc_matmul_kernel!(i8, i16, I16, matmul_i8_i16))
        }
        (DType::I32, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(acc_matmul_kernel!(i8, i32, I32, matmul_i8_i32))
        }
        (DType::I64, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_matmul_kernel!(i8, i64, I64, matmul_i8_i64))
        }
        (DType::I32, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(acc_matmul_kernel!(i16, i32, I32, matmul_i16_i32))
        }
        (DType::I64, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_matmul_kernel!(i16, i64, I64, matmul_i16_i64))
        }
        (DType::I64, [DType::I32, DType::I32], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_matmul_kernel!(i32, i64, I64, matmul_i32_i64))
        }
        (DType::U16, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            Some(acc_matmul_kernel!(u8, u16, U16, matmul_u8_u16))
        }
        (DType::U32, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(acc_matmul_kernel!(u8, u32, U32, matmul_u8_u32))
        }
        (DType::U64, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(acc_matmul_kernel!(u8, u64, U64, matmul_u8_u64))
        }
        (DType::U32, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(acc_matmul_kernel!(u16, u32, U32, matmul_u16_u32))
        }
        (DType::U64, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(acc_matmul_kernel!(u16, u64, U64, matmul_u16_u64))
        }
        (DType::U64, [DType::U32, DType::U32], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(acc_matmul_kernel!(u32, u64, U64, matmul_u32_u64))
        }
        _ => None,
    }
}
