use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::broadcast_shape;
use crate::graph::OpKind;
use crate::ops::vulkan::dispatch::{BindingBytes, VulkanOpSpec};
use crate::ops::vulkan::runtime::VulkanRuntime;
use crate::ops::vulkan::staging_buffer::{return_staging, take_staging, StagingBuffers};
use crate::ops::vulkan::tensor_bytes::{tensor_append_bytes, tensor_byte_len};
use crate::tensor::{DType, TensorValue};
use crate::ops::vulkan::descriptor::{dtype_code, TensorDesc, MAX_DIMS};
use crate::ops::registry::OpMode;
use crate::types::dtype_suffix;

pub struct VulkanIoBuffers {
    pub input_bytes: Vec<u8>,
    pub output_bytes: Vec<u8>,
    pub input0_offset: u32,
    pub input1_offset: u32,
    pub output_offset: u64,
    pub output_alias_input: bool,
}

pub fn build_output_desc(value: &TensorValue, out_rank: usize, byte_offset: u32) -> Result<TensorDesc> {
    if out_rank > MAX_DIMS {
        return Err(anyhow!(
            "vulkan tensors only support up to {} dims (got {})",
            MAX_DIMS,
            out_rank
        ));
    }
    let dtype = value.dtype();
    let mut desc = TensorDesc::default();
    desc.rank = out_rank as u32;
    desc.dtype = dtype_code(dtype);
    desc.elem_bits = dtype.bit_width() as u32;
    desc.byte_offset = byte_offset;
    for i in 0..out_rank {
        desc.shape[i] = value.shape().get(i).copied().unwrap_or(1) as u32;
        desc.strides[i] = value.strides().get(i).copied().unwrap_or(0) as u32;
    }
    Ok(desc)
}

pub fn validate_broadcast_and_rank(
    inputs: &[TensorValue],
    output: &TensorValue,
    max_dims: usize,
) -> Result<(Vec<usize>, bool, bool)> {
    let broadcast = inputs
        .windows(2)
        .any(|pair| pair[0].shape() != pair[1].shape());
    let out_shape = broadcast_shape(inputs[0].shape(), inputs[1].shape())?;
    if output.shape() != out_shape.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match broadcast shape {:?}",
            output.shape(),
            out_shape
        ));
    }
    let exceeds_rank = out_shape.len() > max_dims;
    Ok((out_shape, broadcast, exceeds_rank))
}

pub fn target_name(
    op: OpKind,
    mode: OpMode,
    in_dtype: DType,
    out_dtype: DType,
    use_native_f16: bool,
) -> Result<String> {
    let op_name = op.as_str();
    let in_name = dtype_suffix(in_dtype)?;
    let out_name = dtype_suffix(out_dtype)?;
    if in_dtype.is_packed() {
        return Ok(match mode {
            OpMode::Normal => format!("{op_name}_{in_name}_packed"),
            OpMode::Inplace => format!("{op_name}_{in_name}_packed_inplace"),
            OpMode::Accumulate => format!("{op_name}_{in_name}_accumulate_{out_name}"),
        });
    }
    if in_dtype == DType::F16 {
        let suffix = if use_native_f16 { "native" } else { "simulated" };
        return Ok(match mode {
            OpMode::Normal => format!("{op_name}_{in_name}_normal_{suffix}"),
            OpMode::Inplace => format!("{op_name}_{in_name}_inplace_{suffix}"),
            OpMode::Accumulate => format!("{op_name}_{in_name}_accumulate_{out_name}_{suffix}"),
        });
    }
    Ok(match mode {
        OpMode::Normal => format!("{op_name}_{in_name}_normal"),
        OpMode::Inplace => format!("{op_name}_{in_name}_inplace"),
        OpMode::Accumulate => format!("{op_name}_{in_name}_accumulate_{out_name}"),
    })
}

pub fn prepare_staging_io(
    mode: OpMode,
    inputs: &[TensorValue],
    output: &TensorValue,
    output_dtype: DType,
) -> Result<VulkanIoBuffers> {
    let input0_source = if mode == OpMode::Inplace { output } else { &inputs[0] };
    let output_len = tensor_byte_len(output_dtype, output.len());
    let StagingBuffers {
        input: input_staging,
        output: output_staging,
    } = take_staging()?;
    let mut input_bytes = input_staging;
    let mut output_bytes = output_staging;

    input_bytes.clear();
    tensor_append_bytes(input0_source, &mut input_bytes)?;
    let input0_offset = 0u32;
    let input1_offset = input_bytes.len() as u32;
    tensor_append_bytes(&inputs[1], &mut input_bytes)?;

    output_bytes.clear();
    output_bytes.resize(output_len, 0);
    let output_alias_input = mode == OpMode::Inplace;
    let output_offset = input0_offset as u64;

    Ok(VulkanIoBuffers {
        input_bytes,
        output_bytes,
        input0_offset,
        input1_offset,
        output_offset,
        output_alias_input,
    })
}

pub fn build_desc_bytes(descs: &[crate::ops::vulkan::descriptor::TensorDesc]) -> Vec<u8> {
    bytemuck::cast_slice(descs).to_vec()
}

pub fn dispatch_with_standard_bindings(
    runtime: &VulkanRuntime,
    spec: &VulkanOpSpec<'_>,
    desc_bytes: &[u8],
    input_bytes: &[u8],
    output_bytes: &mut [u8],
    output_alias_input: bool,
    output_offset: u64,
    push_bytes: &[u8],
    dispatch_len: u32,
) -> Result<()> {
    let mut bindings = [
        BindingBytes::ReadOnly(desc_bytes),
        BindingBytes::ReadOnly(input_bytes),
        if output_alias_input {
            BindingBytes::Alias {
                source_binding: 1,
                offset: output_offset,
                bytes: output_bytes,
            }
        } else {
            BindingBytes::ReadWrite(output_bytes)
        },
    ];
    runtime.dispatch_compute(spec, &mut bindings, push_bytes, dispatch_len)
}

pub fn return_staging_buffers(input_bytes: Vec<u8>, output_bytes: Vec<u8>) -> Result<()> {
    return_staging(StagingBuffers {
        input: input_bytes,
        output: output_bytes,
    })?;
    Ok(())
}
