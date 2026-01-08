use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes;
use crate::backend::VulkanBuffer;
use crate::graph::{AttrValue, OpAttrs};
use crate::graph::OpKind;
use crate::tensor::{compute_strides, DType};
use crate::timer::Timer;

pub mod registry;

pub fn relu_generic(attrs: &OpAttrs, a: &VulkanBuffer, thread_id: usize) -> Result<VulkanBuffer> {
    let (negative_slope, clamp_max) = match attrs {
        OpAttrs::Relu {
            negative_slope,
            clamp_max,
        } => (attr_value_f32(negative_slope)?, attr_value_f32(clamp_max)?),
        _ => return Err(anyhow!("relu op expects relu attributes")),
    };
    let runtime = super::runtime_from_buffers(a, None)?;
    let target = super::spv_target_name(OpKind::Relu, a.dtype, attrs)?;
    let entry = super::entry_point_name();
    let output_size = storage_size_bytes(a.dtype) * a.len;
    let output_inner = runtime.create_buffer(output_size)?;
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for relu op", target))?;
    let push = [a.len as u32, negative_slope.to_bits(), clamp_max.to_bits(), 0];
    let duration_ns = runtime.dispatch(
        OpKind::Relu,
        a.dtype,
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
        len: a.len,
        shape: a.shape.clone(),
        strides: compute_strides(a.shape.as_slice()),
        shader: a.shader.clone(),
        inner: output_inner,
    })
}

pub(crate) fn spv_target_name_relu(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::F32, &OpAttrs::Relu { .. }) => Ok("relu_f32".to_string()),
        (DType::I8, &OpAttrs::Relu { .. }) => Ok("relu_i8".to_string()),
        (DType::I16, &OpAttrs::Relu { .. }) => Ok("relu_i16".to_string()),
        (DType::I32, &OpAttrs::Relu { .. }) => Ok("relu_i32".to_string()),
        (DType::I64, &OpAttrs::Relu { .. }) => Ok("relu_i64".to_string()),
        (DType::U8, &OpAttrs::Relu { .. }) => Ok("relu_u8".to_string()),
        (DType::U16, &OpAttrs::Relu { .. }) => Ok("relu_u16".to_string()),
        (DType::U32, &OpAttrs::Relu { .. }) => Ok("relu_u32".to_string()),
        (DType::U64, &OpAttrs::Relu { .. }) => Ok("relu_u64".to_string()),
        (DType::Bool, &OpAttrs::Relu { .. }) => Ok("relu_bool".to_string()),
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for relu dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}

fn attr_value_f32(value: &AttrValue) -> Result<f32> {
    match value {
        AttrValue::Literal(val) => Ok(*val),
        AttrValue::Var(name) => Err(anyhow!("relu op attrs must be resolved: {}", name)),
    }
}
