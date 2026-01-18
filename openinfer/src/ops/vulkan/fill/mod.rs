use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes_for_len;
use crate::backend::VulkanBuffer;
use crate::graph::OpAttrs;
use crate::graph::OpKind;
use crate::tensor::{compute_strides, DType};
use crate::timer::Timer;

pub mod registry;

pub fn fill_generic(
    attrs: &OpAttrs,
    a: &VulkanBuffer,
    thread_id: usize,
) -> Result<VulkanBuffer> {
    let runtime = super::runtime_from_buffers(a, None)?;
    let target = super::spv_target_name(OpKind::Fill, a.effective_dtype, attrs)?;
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for fill op", target))?;
    let output_size = storage_size_bytes_for_len(a.effective_dtype, a.len);
    let output_inner = runtime.create_buffer(output_size)?;
    let value_bits = fill_value_bits(a.effective_dtype, attrs)?;
    let push = [a.len as u32, value_bits, 0, 0];
    let duration_ns = runtime.dispatch(
        OpKind::Fill,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &a.inner,
        &output_inner,
        push,
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

fn fill_value_bits(dtype: DType, attrs: &OpAttrs) -> Result<u32> {
    let value = match attrs {
        OpAttrs::Fill { value } => value,
        _ => return Err(anyhow!("fill expects value attribute")),
    };
    let value = match (dtype, value) {
        (DType::F16 | DType::F32 | DType::F64, crate::graph::AttrValue::Float(val)) => *val,
        (DType::I8 | DType::I16 | DType::I32 | DType::I64, crate::graph::AttrValue::Int(val)) => {
            *val as f32
        }
        (
            DType::U8 | DType::U16 | DType::U32 | DType::U64,
            crate::graph::AttrValue::UInt(val),
        ) => *val as f32,
        (
            DType::U8 | DType::U16 | DType::U32 | DType::U64,
            crate::graph::AttrValue::Int(val),
        ) => {
            if *val < 0 {
                return Err(anyhow!("fill expects unsigned value"));
            }
            *val as f32
        }
        (DType::Bool, crate::graph::AttrValue::Bool(val)) => {
            if *val { 1.0 } else { 0.0 }
        }
        (_, crate::graph::AttrValue::Var(_)) => {
            return Err(anyhow!("fill expects resolved value"));
        }
        _ => {
            return Err(anyhow!(
                "fill value dtype mismatch for {:?}",
                dtype
            ))
        }
    };
    Ok(value.to_bits())
}

pub(crate) fn spv_target_name_fill(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::I8, &OpAttrs::Fill { .. })
        | (DType::I16, &OpAttrs::Fill { .. })
        | (DType::F32, &OpAttrs::Fill { .. })
        | (DType::Bool, &OpAttrs::Fill { .. })
        | (DType::U8, &OpAttrs::Fill { .. })
        | (DType::U16, &OpAttrs::Fill { .. })
        | (DType::I32, &OpAttrs::Fill { .. })
        | (DType::U32, &OpAttrs::Fill { .. })
        | (DType::I64, &OpAttrs::Fill { .. })
        | (DType::U64, &OpAttrs::Fill { .. }) => {
            Ok(format!("fill_{}", super::dtype_suffix(dtype).unwrap()))
        }
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for fill dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}
