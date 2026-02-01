use anyhow::{anyhow, Result};

use crate::graph::{OpAttrs, OpKind};
use crate::ops::cpu::reduce::{axes_from_attrs, keepdims_from_attrs, output_shape};
use crate::ops::registry::{op_supports_dtype, OpKey, OpMode};
use crate::tensor::{DType, TensorValue};

use bytemuck::{Pod, Zeroable};

use crate::ops::vulkan::descriptor::{build_tensor_desc, MAX_DIMS};
use crate::ops::vulkan::dispatch::VulkanOpSpec;
use crate::ops::vulkan::op_helpers::{
    build_desc_bytes, build_output_desc, dispatch_with_standard_bindings, prepare_unary_staging_io,
    return_staging_buffers, target_name,
};
use crate::ops::vulkan::runtime::get_vulkan_runtime;
use crate::ops::vulkan::tensor_bytes::write_tensor_from_bytes;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct MinAxisPush {
    input_len: u32,
    output_len: u32,
    input_rank: u32,
    output_rank: u32,
    axes_mask: u32,
    keepdims: u32,
    _pad0: u32,
    _pad1: u32,
}

pub fn min_axis_normal_dispatch(
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    dispatch_min_axis(OpMode::Normal, attrs, inputs, output)
}

fn dispatch_min_axis(
    mode: OpMode,
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    crate::vk_trace!(
        "dispatch_min_axis mode={:?} input_dtype={:?} output_dtype={:?}",
        mode,
        inputs.get(0).map(|t| t.dtype()),
        output.as_ref().map(|t| t.dtype())
    );
    if inputs.len() != 1 {
        return Err(anyhow!("min_axis expects 1 input, got {}", inputs.len()));
    }
    let output = output.ok_or_else(|| anyhow!("min_axis requires an output tensor"))?;
    let input_dtype = inputs[0].dtype();
    let output_dtype = output.dtype();

    let axes = axes_from_attrs(attrs, inputs[0].shape().len())?;
    let keepdims = keepdims_from_attrs(attrs);
    let expected_shape = output_shape(inputs[0].shape(), &axes, keepdims);
    if output.shape() != expected_shape.as_slice() {
        return Err(anyhow!(
            "min_axis output shape {:?} does not match expected shape {:?}",
            output.shape(),
            expected_shape
        ));
    }
    let in_rank = inputs[0].shape().len();
    let out_rank = output.shape().len();
    if in_rank > MAX_DIMS || out_rank > MAX_DIMS {
        return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype);
    }

    let runtime = match get_vulkan_runtime() {
        Some(runtime) => runtime,
        None => {
            crate::vk_trace!("vulkan runtime not initialized, falling back to cpu");
            return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype);
        }
    };

    if !runtime.caps().supports_dtype(input_dtype) || !runtime.caps().supports_dtype(output_dtype) {
        crate::vk_trace!(
            "dtype unsupported by vulkan caps (input={:?}, output={:?}), cpu fallback",
            input_dtype,
            output_dtype
        );
        return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype);
    }
    if !op_supports_dtype(OpKind::MinAxis, mode, input_dtype, output_dtype) {
        crate::vk_trace!(
            "vulkan target unsupported (mode={:?}, in={:?}, out={:?}), cpu fallback",
            mode,
            input_dtype,
            output_dtype
        );
        return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype);
    }

    let mut io_buffers =
        prepare_unary_staging_io(OpMode::Normal, &inputs[0], output, output_dtype)?;

    let mut descs = Vec::with_capacity(2);
    descs.push(build_tensor_desc(&inputs[0], in_rank, io_buffers.input_offset)?);
    descs.push(build_output_desc(output, out_rank, 0)?);

    let mut axes_mask = 0u32;
    for axis in &axes {
        axes_mask |= 1u32 << axis;
    }

    let push = MinAxisPush {
        input_len: inputs[0].len() as u32,
        output_len: output.len() as u32,
        input_rank: in_rank as u32,
        output_rank: out_rank as u32,
        axes_mask,
        keepdims: if keepdims { 1 } else { 0 },
        _pad0: 0,
        _pad1: 0,
    };

    let target = match target_name(OpKind::MinAxis, mode, input_dtype, output_dtype) {
        Ok(name) => name,
        Err(_) => {
            crate::vk_trace!("target name not found, cpu fallback");
            return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype);
        }
    };
    crate::vk_trace!("dispatching vulkan min_axis target={}", target);
    let desc_bytes = build_desc_bytes(&descs);
    let push_bytes = bytemuck::bytes_of(&push).to_vec();
    let spec = VulkanOpSpec {
        entry: &target,
        spv_dir: "src/ops/vulkan/min_axis/bin",
        workgroup_size: [256, 1, 1],
        push_constant_size: std::mem::size_of::<MinAxisPush>() as u32,
    };
    let dispatched = dispatch_with_standard_bindings(
        runtime,
        &spec,
        desc_bytes.as_slice(),
        io_buffers.input_bytes.as_slice(),
        io_buffers.output_bytes.as_mut_slice(),
        false,
        0,
        &push_bytes,
        output.len() as u32,
    );
    if let Err(err) = &dispatched {
        crate::vk_trace!("vulkan dispatch error: {}", err);
    }
    if dispatched.is_ok() {
        write_tensor_from_bytes(output, &io_buffers.output_bytes)?;
        crate::vk_trace!("vulkan dispatch successful");
        return_staging_buffers(io_buffers.input_bytes, io_buffers.output_bytes)?;
        return Ok(());
    }
    crate::vk_trace!("vulkan dispatch failed, cpu fallback");
    return_staging_buffers(io_buffers.input_bytes, io_buffers.output_bytes)?;

    cpu_fallback(mode, attrs, inputs, Some(output), output_dtype)
}

fn cpu_fallback(
    mode: OpMode,
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
    output_dtype: DType,
) -> Result<()> {
    let key = OpKey {
        kind: OpKind::MinAxis,
        mode,
        broadcast: false,
        inputs: inputs.iter().map(|tensor| tensor.dtype()).collect(),
        out0: output_dtype,
    };
    let kernel = crate::ops::cpu::registry::lookup_kernel(key)?;
    kernel(attrs, inputs, output)
}
