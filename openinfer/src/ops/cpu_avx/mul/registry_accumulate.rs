use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, I2, I4, U2, U4, Tensor, TensorElement, TensorOptions, TensorValue};

use super::{
    mul_i16_i32, mul_i16_i64, mul_i32_i64, mul_i8_i16, mul_i8_i32, mul_i8_i64, mul_u16_u32,
    mul_u16_u64, mul_u32_u64, mul_u8_u16, mul_u8_u32, mul_u8_u64,
    mul_i4_i8_packed, mul_i4_i16_packed, mul_i4_i32_packed, mul_i4_i64_packed,
    mul_i2_i8_packed, mul_i2_i16_packed, mul_i2_i32_packed, mul_i2_i64_packed,
    mul_u4_u8_packed, mul_u4_u16_packed, mul_u4_u32_packed, mul_u4_u64_packed,
    mul_u2_u8_packed, mul_u2_u16_packed, mul_u2_u32_packed, mul_u2_u64_packed,
};

pub fn lookup_kernel_cpu_avx_mul_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    fn tensor_from_vec<T: TensorElement>(data: Vec<T>, shape: &[usize]) -> anyhow::Result<Tensor<T>> {
        Tensor::from_vec_with_opts(
            data,
            TensorOptions {
                shape: Some(shape.to_vec()),
                ..TensorOptions::default()
            },
        )
    }

    macro_rules! acc_binary_kernel_slice {
        ($in:ty, $out_ty:ty, $out_variant:ident, $func:ident) => {
            KernelFn::Host(Box::new(|_attrs, inputs, output, thread_id| {
                if inputs.len() < 2 {
                    return Err(anyhow::anyhow!("mul op expects 2 inputs"));
                }
                let a = <$in as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("mul input 0 dtype mismatch"))?;
                let b = <$in as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("mul input 1 dtype mismatch"))?;
                let out_slice = output.and_then(|out| match out {
                    TensorValue::$out_variant(t) => Some(&mut t.data),
                    _ => None,
                });
                let out = $func(
                    &a.data,
                    &b.data,
                    out_slice.map(|out| out.as_mut_slice()),
                    thread_id,
                )?;
                match out {
                    Some(out) => {
                        let tensor = tensor_from_vec(out, a.shape())?;
                        Ok(Some(<$out_ty as TensorElement>::into_value(tensor)))
                    }
                    None => Ok(None),
                }
            }))
        };
    }

    macro_rules! acc_binary_kernel_packed {
        ($in:ty, $out_ty:ty, $out_variant:ident, $func:ident) => {
            KernelFn::Host(Box::new(|_attrs, inputs, output, thread_id| {
                if inputs.len() < 2 {
                    return Err(anyhow::anyhow!("mul op expects 2 inputs"));
                }
                let a = <$in as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("mul input 0 dtype mismatch"))?;
                let b = <$in as TensorElement>::from_value(&inputs[1])
                    .ok_or_else(|| anyhow::anyhow!("mul input 1 dtype mismatch"))?;
                let out_slice = output.and_then(|out| match out {
                    TensorValue::$out_variant(t) => Some(&mut t.data),
                    _ => None,
                });
                let out = $func(
                    &a.data,
                    &b.data,
                    a.numel(),
                    out_slice.map(|out| out.as_mut_slice()),
                    thread_id,
                )?;
                match out {
                    Some(out) => {
                        let tensor = tensor_from_vec(out, a.shape())?;
                        Ok(Some(<$out_ty as TensorElement>::into_value(tensor)))
                    }
                    None => Ok(None),
                }
            }))
        };
    }
    match (output_dtype, input_dtypes, attrs) {
        (DType::I16, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(acc_binary_kernel_slice!(i8, i16, I16, mul_i8_i16))
        }
        (DType::I32, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(acc_binary_kernel_slice!(i8, i32, I32, mul_i8_i32))
        }
        (DType::I64, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_binary_kernel_slice!(i8, i64, I64, mul_i8_i64))
        }
        (DType::I32, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(acc_binary_kernel_slice!(i16, i32, I32, mul_i16_i32))
        }
        (DType::I64, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_binary_kernel_slice!(i16, i64, I64, mul_i16_i64))
        }
        (DType::I64, [DType::I32, DType::I32], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_binary_kernel_slice!(i32, i64, I64, mul_i32_i64))
        }
        (DType::U16, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            Some(acc_binary_kernel_slice!(u8, u16, U16, mul_u8_u16))
        }
        (DType::U32, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(acc_binary_kernel_slice!(u8, u32, U32, mul_u8_u32))
        }
        (DType::U64, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(acc_binary_kernel_slice!(u8, u64, U64, mul_u8_u64))
        }
        (DType::U32, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(acc_binary_kernel_slice!(u16, u32, U32, mul_u16_u32))
        }
        (DType::U64, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(acc_binary_kernel_slice!(u16, u64, U64, mul_u16_u64))
        }
        (DType::U64, [DType::U32, DType::U32], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(acc_binary_kernel_slice!(u32, u64, U64, mul_u32_u64))
        }
        (DType::I8, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            Some(acc_binary_kernel_packed!(I4, i8, I8, mul_i4_i8_packed))
        }
        (DType::I16, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(acc_binary_kernel_packed!(I4, i16, I16, mul_i4_i16_packed))
        }
        (DType::I32, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(acc_binary_kernel_packed!(I4, i32, I32, mul_i4_i32_packed))
        }
        (DType::I64, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_binary_kernel_packed!(I4, i64, I64, mul_i4_i64_packed))
        }
        (DType::I8, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            Some(acc_binary_kernel_packed!(I2, i8, I8, mul_i2_i8_packed))
        }
        (DType::I16, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(acc_binary_kernel_packed!(I2, i16, I16, mul_i2_i16_packed))
        }
        (DType::I32, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(acc_binary_kernel_packed!(I2, i32, I32, mul_i2_i32_packed))
        }
        (DType::I64, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_binary_kernel_packed!(I2, i64, I64, mul_i2_i64_packed))
        }
        (DType::U8, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U8 }) => {
            Some(acc_binary_kernel_packed!(U4, u8, U8, mul_u4_u8_packed))
        }
        (DType::U16, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            Some(acc_binary_kernel_packed!(U4, u16, U16, mul_u4_u16_packed))
        }
        (DType::U32, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(acc_binary_kernel_packed!(U4, u32, U32, mul_u4_u32_packed))
        }
        (DType::U64, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(acc_binary_kernel_packed!(U4, u64, U64, mul_u4_u64_packed))
        }
        (DType::U8, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U8 }) => {
            Some(acc_binary_kernel_packed!(U2, u8, U8, mul_u2_u8_packed))
        }
        (DType::U16, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            Some(acc_binary_kernel_packed!(U2, u16, U16, mul_u2_u16_packed))
        }
        (DType::U32, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            Some(acc_binary_kernel_packed!(U2, u32, U32, mul_u2_u32_packed))
        }
        (DType::U64, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            Some(acc_binary_kernel_packed!(U2, u64, U64, mul_u2_u64_packed))
        }
        _ => None,
    }
}
