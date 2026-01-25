use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes_for_len;
use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::graph::OpKind;
use crate::tensor::{compute_strides, DType};
use crate::timer::Timer;

pub mod registry;
pub mod registry_accumulate;
pub mod registry_inplace;

#[repr(C)]
struct FillPushConstsFloat {
    len: u32,
    value_f32: f32,
    value_f64: f64,
}

#[repr(C)]
struct FillPushConstsSigned {
    len: u32,
    value_i32: i32,
    value_i64: i64,
}

#[repr(C)]
struct FillPushConstsUnsigned {
    len: u32,
    value_u32: u32,
    value_u64: u64,
}

enum FillPushConsts {
    Float(FillPushConstsFloat),
    Signed(FillPushConstsSigned),
    Unsigned(FillPushConstsUnsigned),
}

impl FillPushConsts {
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            match self {
                FillPushConsts::Float(value) => std::slice::from_raw_parts(
                    (value as *const FillPushConstsFloat).cast::<u8>(),
                    std::mem::size_of::<FillPushConstsFloat>(),
                ),
                FillPushConsts::Signed(value) => std::slice::from_raw_parts(
                    (value as *const FillPushConstsSigned).cast::<u8>(),
                    std::mem::size_of::<FillPushConstsSigned>(),
                ),
                FillPushConsts::Unsigned(value) => std::slice::from_raw_parts(
                    (value as *const FillPushConstsUnsigned).cast::<u8>(),
                    std::mem::size_of::<FillPushConstsUnsigned>(),
                ),
            }
        }
    }

    fn as_u32s(&self) -> [u32; 4] {
        let bytes = self.as_bytes();
        let mut out = [0u32; 4];
        for (idx, chunk) in bytes.chunks_exact(4).take(4).enumerate() {
            out[idx] = u32::from_ne_bytes(chunk.try_into().unwrap());
        }
        out
    }
}

pub fn fill_generic(
    attrs: &OpAttrs,
    a: &VulkanBuffer,
    thread_id: usize,
) -> Result<VulkanBuffer> {
    let runtime = super::runtime_from_buffers(a, None)?;
    let target = if a.effective_dtype == DType::F16 && runtime.use_native_f16() {
        "fill_f16_native".to_string()
    } else {
        super::spv_target_name(OpKind::Fill, a.effective_dtype, attrs)?
    };
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for fill op", target))?;
    let output_size = storage_size_bytes_for_len(a.effective_dtype, a.len);
    let output_inner = runtime.create_buffer(output_size)?;
    let push = fill_push_consts(a.effective_dtype, attrs, a.len)?;
    let push = push.as_u32s();
    let duration_ns = runtime.dispatch(
        OpKind::Fill,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &a.inner,
        &output_inner,
        &push,
        a.len,
    )?;
    Timer::record(thread_id, duration_ns);
    Ok(VulkanBuffer {
        dtype: a.dtype,
        effective_dtype: a.effective_dtype,
        len: a.len,
        shape: a.shape.clone(),
        strides: compute_strides(a.shape.as_slice()),
        shader: a.shader.clone(),
        inner: output_inner,
    })
}

pub fn fill_inplace_generic(
    attrs: &OpAttrs,
    a: &VulkanBuffer,
    thread_id: usize,
) -> Result<VulkanBuffer> {
    let runtime = super::runtime_from_buffers(a, None)?;
    let target = if a.effective_dtype == DType::F16 && runtime.use_native_f16() {
        "fill_inplace_f16_native".to_string()
    } else {
        spv_target_name_fill_inplace(a.effective_dtype, attrs)?
    };
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for fill inplace", target))?;
    let output_size = storage_size_bytes_for_len(a.effective_dtype, a.len);
    if output_size > a.inner.size as usize {
        return Err(anyhow!("fill inplace output buffer too small"));
    }
    let push = fill_push_consts(a.effective_dtype, attrs, a.len)?;
    let push = push.as_u32s();
    let duration_ns = runtime.dispatch(
        OpKind::Fill,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &a.inner,
        &a.inner,
        &push,
        a.len,
    )?;
    Timer::record(thread_id, duration_ns);
    Ok(VulkanBuffer {
        dtype: a.dtype,
        effective_dtype: a.effective_dtype,
        len: a.len,
        shape: a.shape.clone(),
        strides: compute_strides(a.shape.as_slice()),
        shader: a.shader.clone(),
        inner: a.inner.clone(),
    })
}

fn fill_push_consts(dtype: DType, attrs: &OpAttrs, len: usize) -> Result<FillPushConsts> {
    let value = match attrs {
        OpAttrs::Fill { value } => value,
        _ => return Err(anyhow!("fill expects value attribute")),
    };
    match dtype {
        DType::F16 | DType::BF16 | DType::F8E5M2 | DType::F32 => {
            let value = fill_attr_f32(value)?;
            Ok(FillPushConsts::Float(FillPushConstsFloat {
                len: len as u32,
                value_f32: value,
                value_f64: 0.0,
            }))
        }
        DType::F64 => {
            let value = fill_attr_f64(value)?;
            Ok(FillPushConsts::Float(FillPushConstsFloat {
                len: len as u32,
                value_f32: 0.0,
                value_f64: value,
            }))
        }
        DType::I8 | DType::I16 | DType::I32 | DType::I4 | DType::I2 | DType::I1 => {
            let value = fill_attr_i64(value)?;
            Ok(FillPushConsts::Signed(FillPushConstsSigned {
                len: len as u32,
                value_i32: value as i32,
                value_i64: value,
            }))
        }
        DType::I64 => {
            let value = fill_attr_i64(value)?;
            Ok(FillPushConsts::Signed(FillPushConstsSigned {
                len: len as u32,
                value_i32: value as i32,
                value_i64: value,
            }))
        }
        DType::U8 | DType::U16 | DType::U32 | DType::U4 | DType::U2 | DType::U1 | DType::Bitset => {
            let value = fill_attr_u64(value)?;
            Ok(FillPushConsts::Unsigned(FillPushConstsUnsigned {
                len: len as u32,
                value_u32: value as u32,
                value_u64: value,
            }))
        }
        DType::U64 => {
            let value = fill_attr_u64(value)?;
            Ok(FillPushConsts::Unsigned(FillPushConstsUnsigned {
                len: len as u32,
                value_u32: value as u32,
                value_u64: value,
            }))
        }
        DType::Bool => {
            let value = fill_attr_bool(value)?;
            let value_u32 = if value { 1 } else { 0 };
            Ok(FillPushConsts::Unsigned(FillPushConstsUnsigned {
                len: len as u32,
                value_u32,
                value_u64: value_u32 as u64,
            }))
        }
        _ => Err(anyhow!("fill value dtype mismatch for {:?}", dtype)),
    }
}

fn fill_attr_f32(value: &crate::graph::AttrValue) -> Result<f32> {
    match value {
        crate::graph::AttrValue::Float(val) => Ok(*val),
        crate::graph::AttrValue::Double(val) => {
            if val.is_finite() && val.abs() > f32::MAX as f64 {
                return Err(anyhow!("fill value {} is out of range for f32", val));
            }
            Ok(*val as f32)
        }
        _ => Err(anyhow!("fill expects f32 value")),
    }
}

fn fill_attr_f64(value: &crate::graph::AttrValue) -> Result<f64> {
    match value {
        crate::graph::AttrValue::Float(val) => Ok(*val as f64),
        crate::graph::AttrValue::Double(val) => Ok(*val),
        _ => Err(anyhow!("fill expects f64 value")),
    }
}

fn fill_attr_i64(value: &crate::graph::AttrValue) -> Result<i64> {
    match value {
        crate::graph::AttrValue::Int(val) => Ok(*val),
        _ => Err(anyhow!("fill expects i64 value")),
    }
}

fn fill_attr_u64(value: &crate::graph::AttrValue) -> Result<u64> {
    match value {
        crate::graph::AttrValue::UInt(val) => Ok(*val),
        crate::graph::AttrValue::Int(val) if *val >= 0 => Ok(*val as u64),
        _ => Err(anyhow!("fill expects unsigned value")),
    }
}

fn fill_attr_bool(value: &crate::graph::AttrValue) -> Result<bool> {
    match value {
        crate::graph::AttrValue::Bool(val) => Ok(*val),
        _ => Err(anyhow!("fill expects bool value")),
    }
}

pub(crate) fn spv_target_name_fill(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::I8, &OpAttrs::Fill { .. })
        | (DType::I16, &OpAttrs::Fill { .. })
        | (DType::I32, &OpAttrs::Fill { .. })
        | (DType::I64, &OpAttrs::Fill { .. })
        | (DType::U8, &OpAttrs::Fill { .. })
        | (DType::U16, &OpAttrs::Fill { .. })
        | (DType::U32, &OpAttrs::Fill { .. })
        | (DType::U64, &OpAttrs::Fill { .. })
        | (DType::I4, &OpAttrs::Fill { .. })
        | (DType::I2, &OpAttrs::Fill { .. })
        | (DType::I1, &OpAttrs::Fill { .. })
        | (DType::U4, &OpAttrs::Fill { .. })
        | (DType::U2, &OpAttrs::Fill { .. })
        | (DType::U1, &OpAttrs::Fill { .. })
        | (DType::Bool, &OpAttrs::Fill { .. })
        | (DType::Bitset, &OpAttrs::Fill { .. })
        | (DType::F16, &OpAttrs::Fill { .. })
        | (DType::BF16, &OpAttrs::Fill { .. })
        | (DType::F8E5M2, &OpAttrs::Fill { .. })
        | (DType::F32, &OpAttrs::Fill { .. })
        | (DType::F64, &OpAttrs::Fill { .. }) => {
            Ok(format!("fill_{}", super::dtype_suffix(dtype).unwrap()))
        }
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for fill dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}

pub(crate) fn spv_target_name_fill_inplace(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::I8, &OpAttrs::Fill { .. })
        | (DType::I16, &OpAttrs::Fill { .. })
        | (DType::I32, &OpAttrs::Fill { .. })
        | (DType::I64, &OpAttrs::Fill { .. })
        | (DType::U8, &OpAttrs::Fill { .. })
        | (DType::U16, &OpAttrs::Fill { .. })
        | (DType::U32, &OpAttrs::Fill { .. })
        | (DType::U64, &OpAttrs::Fill { .. })
        | (DType::I4, &OpAttrs::Fill { .. })
        | (DType::I2, &OpAttrs::Fill { .. })
        | (DType::I1, &OpAttrs::Fill { .. })
        | (DType::U4, &OpAttrs::Fill { .. })
        | (DType::U2, &OpAttrs::Fill { .. })
        | (DType::U1, &OpAttrs::Fill { .. })
        | (DType::Bool, &OpAttrs::Fill { .. })
        | (DType::Bitset, &OpAttrs::Fill { .. })
        | (DType::F16, &OpAttrs::Fill { .. })
        | (DType::BF16, &OpAttrs::Fill { .. })
        | (DType::F8E5M2, &OpAttrs::Fill { .. })
        | (DType::F32, &OpAttrs::Fill { .. })
        | (DType::F64, &OpAttrs::Fill { .. }) => {
            Ok(format!("fill_inplace_{}", super::dtype_suffix(dtype).unwrap()))
        }
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for fill inplace dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}
