use anyhow::anyhow;

use crate::graph::{AttrValue, OpAttrs};
use crate::ops::registry::{HostInplaceKernel, InplaceKernelFn};
use crate::tensor::TensorValue;
use crate::tensor::{DType, BF16, F16, F8E5M2};
use super::{fill_i4, fill_i2, fill_i1, fill_u4, fill_u2, fill_u1};

#[allow(dead_code)]
pub fn supports_fill_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!((output_dtype, input_dtypes, attrs), (_, [_], OpAttrs::Fill { .. }))
}

pub fn lookup_kernel_cpu_fill_inplace(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    if !supports_fill_inplace(output_dtype, input_dtypes, attrs) {
        return None;
    }
    let kernel: HostInplaceKernel = Box::new(|attrs, output, _inputs, thread_id| {
        let value = match attrs {
            OpAttrs::Fill { value } => value.clone(),
            _ => return Err(anyhow!("fill inplace expects fill attributes")),
        };
        match (output, value) {
            (TensorValue::I8(out), AttrValue::Int(val)) => {
                out.data.fill(val as i8);
                Ok(())
            }
            (TensorValue::I16(out), AttrValue::Int(val)) => {
                out.data.fill(val as i16);
                Ok(())
            }
            (TensorValue::I32(out), AttrValue::Int(val)) => {
                out.data.fill(val as i32);
                Ok(())
            }
            (TensorValue::I64(out), AttrValue::Int(val)) => {
                out.data.fill(val as i64);
                Ok(())
            }
            (TensorValue::U8(out), AttrValue::UInt(val)) => {
                out.data.fill(val as u8);
                Ok(())
            }
            (TensorValue::U8(out), AttrValue::Int(val)) => {
                if val < 0 {
                    return Err(anyhow!("fill expects u8 value"));
                }
                out.data.fill(val as u8);
                Ok(())
            }
            (TensorValue::U16(out), AttrValue::UInt(val)) => {
                out.data.fill(val as u16);
                Ok(())
            }
            (TensorValue::U16(out), AttrValue::Int(val)) => {
                if val < 0 {
                    return Err(anyhow!("fill expects u16 value"));
                }
                out.data.fill(val as u16);
                Ok(())
            }
            (TensorValue::U32(out), AttrValue::UInt(val)) => {
                out.data.fill(val as u32);
                Ok(())
            }
            (TensorValue::U32(out), AttrValue::Int(val)) => {
                if val < 0 {
                    return Err(anyhow!("fill expects u32 value"));
                }
                out.data.fill(val as u32);
                Ok(())
            }
            (TensorValue::U64(out), AttrValue::UInt(val)) => {
                out.data.fill(val as u64);
                Ok(())
            }
            (TensorValue::U64(out), AttrValue::Int(val)) => {
                if val < 0 {
                    return Err(anyhow!("fill expects u64 value"));
                }
                out.data.fill(val as u64);
                Ok(())
            }
            (TensorValue::F32(out), AttrValue::Float(val)) => {
                out.data.fill(val);
                Ok(())
            }
            (TensorValue::F64(out), AttrValue::Float(val)) => {
                out.data.fill(val as f64);
                Ok(())
            }
            (TensorValue::F16(out), AttrValue::Float(val)) => {
                out.data.fill(F16::from_f32(val));
                Ok(())
            }
            (TensorValue::BF16(out), AttrValue::Float(val)) => {
                out.data.fill(BF16::from_f32(val));
                Ok(())
            }
            (TensorValue::F8E5M2(out), AttrValue::Float(val)) => {
                out.data.fill(F8E5M2::from_f32(val));
                Ok(())
            }
            (TensorValue::Bool(out), AttrValue::Bool(val)) => {
                out.data.fill(val);
                Ok(())
            }
            (TensorValue::Bitset(out), AttrValue::UInt(val)) => {
                out.data.fill(crate::tensor::Bitset { bits: val as u8 });
                Ok(())
            }
            (TensorValue::Bitset(out), AttrValue::Int(val)) => {
                if val < 0 {
                    return Err(anyhow!("fill expects bitset value"));
                }
                out.data.fill(crate::tensor::Bitset { bits: val as u8 });
                Ok(())
            }
            (TensorValue::I4(out), AttrValue::Int(_)) => {
                out.data = fill_i4(attrs, out.numel(), thread_id)?;
                Ok(())
            }
            (TensorValue::I2(out), AttrValue::Int(_)) => {
                out.data = fill_i2(attrs, out.numel(), thread_id)?;
                Ok(())
            }
            (TensorValue::I1(out), AttrValue::Int(_)) => {
                out.data = fill_i1(attrs, out.numel(), thread_id)?;
                Ok(())
            }
            (TensorValue::U4(out), AttrValue::UInt(_)) => {
                out.data = fill_u4(attrs, out.numel(), thread_id)?;
                Ok(())
            }
            (TensorValue::U4(out), AttrValue::Int(val)) => {
                if val < 0 {
                    return Err(anyhow!("fill expects u4 value"));
                }
                out.data = fill_u4(attrs, out.numel(), thread_id)?;
                Ok(())
            }
            (TensorValue::U2(out), AttrValue::UInt(_)) => {
                out.data = fill_u2(attrs, out.numel(), thread_id)?;
                Ok(())
            }
            (TensorValue::U2(out), AttrValue::Int(val)) => {
                if val < 0 {
                    return Err(anyhow!("fill expects u2 value"));
                }
                out.data = fill_u2(attrs, out.numel(), thread_id)?;
                Ok(())
            }
            (TensorValue::U1(out), AttrValue::UInt(_)) => {
                out.data = fill_u1(attrs, out.numel(), thread_id)?;
                Ok(())
            }
            (TensorValue::U1(out), AttrValue::Int(val)) => {
                if val < 0 {
                    return Err(anyhow!("fill expects u1 value"));
                }
                out.data = fill_u1(attrs, out.numel(), thread_id)?;
                Ok(())
            }
            _ => Err(anyhow!("fill inplace dtype mismatch")),
        }
    });
    Some(InplaceKernelFn::Host(kernel))
}
