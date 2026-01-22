use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, Tensor, TensorElement, TensorOptions, TensorValue};

use super::{
    abs_i16_i32, abs_i16_i64, abs_i32_i64, abs_i8_i16, abs_i8_i32, abs_i8_i64,
    abs_i4_i8_packed, abs_i4_i16_packed, abs_i4_i32_packed, abs_i4_i64_packed,
    abs_i2_i8_packed, abs_i2_i16_packed, abs_i2_i32_packed, abs_i2_i64_packed,
};
use crate::tensor::{I2, I4};

pub fn lookup_kernel_cpu_avx2_abs_accumulate(
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

    macro_rules! acc_unary_kernel_slice {
        ($in:ty, $out_ty:ty, $out_variant:ident, $func:ident) => {
            KernelFn::Host(Box::new(|_attrs, inputs, output, thread_id| {
                if inputs.is_empty() {
                    return Err(anyhow::anyhow!("abs op expects 1 input"));
                }
                let a = <$in as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out_slice = output.and_then(|out| match out {
                    TensorValue::$out_variant(t) => Some(&mut t.data),
                    _ => None,
                });
                let out = $func(
                    &a.data,
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

    macro_rules! acc_unary_kernel_packed {
        ($in:ty, $out_ty:ty, $out_variant:ident, $func:ident) => {
            KernelFn::Host(Box::new(|_attrs, inputs, output, thread_id| {
                if inputs.is_empty() {
                    return Err(anyhow::anyhow!("abs op expects 1 input"));
                }
                let a = <$in as TensorElement>::from_value(&inputs[0])
                    .ok_or_else(|| anyhow::anyhow!("abs input 0 dtype mismatch"))?;
                let out_slice = output.and_then(|out| match out {
                    TensorValue::$out_variant(t) => Some(&mut t.data),
                    _ => None,
                });
                let out = $func(
                    &a.data,
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
        (DType::I16, [DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(acc_unary_kernel_slice!(i8, i16, I16, abs_i8_i16))
        }
        (DType::I32, [DType::I8], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(acc_unary_kernel_slice!(i8, i32, I32, abs_i8_i32))
        }
        (DType::I64, [DType::I8], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_unary_kernel_slice!(i8, i64, I64, abs_i8_i64))
        }
        (DType::I32, [DType::I16], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(acc_unary_kernel_slice!(i16, i32, I32, abs_i16_i32))
        }
        (DType::I64, [DType::I16], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_unary_kernel_slice!(i16, i64, I64, abs_i16_i64))
        }
        (DType::I64, [DType::I32], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_unary_kernel_slice!(i32, i64, I64, abs_i32_i64))
        }
        (DType::I8, [DType::I4], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            Some(acc_unary_kernel_packed!(I4, i8, I8, abs_i4_i8_packed))
        }
        (DType::I16, [DType::I4], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(acc_unary_kernel_packed!(I4, i16, I16, abs_i4_i16_packed))
        }
        (DType::I32, [DType::I4], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(acc_unary_kernel_packed!(I4, i32, I32, abs_i4_i32_packed))
        }
        (DType::I64, [DType::I4], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_unary_kernel_packed!(I4, i64, I64, abs_i4_i64_packed))
        }
        (DType::I8, [DType::I2], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            Some(acc_unary_kernel_packed!(I2, i8, I8, abs_i2_i8_packed))
        }
        (DType::I16, [DType::I2], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            Some(acc_unary_kernel_packed!(I2, i16, I16, abs_i2_i16_packed))
        }
        (DType::I32, [DType::I2], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            Some(acc_unary_kernel_packed!(I2, i32, I32, abs_i2_i32_packed))
        }
        (DType::I64, [DType::I2], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            Some(acc_unary_kernel_packed!(I2, i64, I64, abs_i2_i64_packed))
        }
        _ => None,
    }
}
