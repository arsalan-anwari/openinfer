use anyhow::{anyhow, Result};

use crate::graph::{OpAttrs, OpKind};
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
struct IsFinitePush {
    len: u32,
    tensor_count: u32,
    params_offset: u32,
    flags: u32,
    input_len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub fn is_finite_normal_dispatch(
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    dispatch_is_finite(OpMode::Normal, attrs, inputs, output)
}

fn dispatch_is_finite(
    mode: OpMode,
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    crate::vk_trace!(
        "dispatch_is_finite mode={:?} input_dtype={:?} output_dtype={:?}",
        mode,
        inputs.get(0).map(|t| t.dtype()),
        output.as_ref().map(|t| t.dtype())
    );
    if inputs.len() != 1 {
        return Err(anyhow!("is_finite expects 1 input, got {}", inputs.len()));
    }
    let output = output.ok_or_else(|| anyhow!("is_finite requires an output tensor"))?;
    let input_dtype = inputs[0].dtype();
    let output_dtype = DType::Bool;
    if output.dtype() != DType::Bool {
        return Err(anyhow!(
            "is_finite requires bool output dtype, got {:?}",
            output.dtype()
        ));
    }
    if !output.shape().is_empty() || output.len() != 1 {
        return Err(anyhow!("is_finite expects a scalar bool output"));
    }
    if inputs[0].shape().len() > MAX_DIMS {
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
    if !op_supports_dtype(OpKind::IsFinite, mode, input_dtype, output_dtype) {
        crate::vk_trace!(
            "vulkan target unsupported (mode={:?}, in={:?}, out={:?}), cpu fallback",
            mode,
            input_dtype,
            output_dtype
        );
        return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype);
    }

    let mut io_buffers =
        prepare_unary_staging_io(mode, &inputs[0], output, output_dtype)?;

    let input_rank = inputs[0].shape().len();
    let mut descs = Vec::with_capacity(2);
    descs.push(build_tensor_desc(&inputs[0], input_rank, io_buffers.input_offset)?);
    descs.push(build_output_desc(output, 0, 0)?);

    let push = IsFinitePush {
        len: output.len() as u32,
        tensor_count: descs.len() as u32,
        params_offset: 0,
        flags: 0,
        input_len: inputs[0].len() as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let target = match target_name(OpKind::IsFinite, mode, input_dtype, output_dtype) {
        Ok(name) => name,
        Err(_) => {
            crate::vk_trace!("target name not found, cpu fallback");
            return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype);
        }
    };
    crate::vk_trace!("dispatching vulkan is_finite target={}", target);
    let desc_bytes = build_desc_bytes(&descs);
    let push_bytes = bytemuck::bytes_of(&push).to_vec();
    let spec = VulkanOpSpec {
        entry: &target,
        spv_dir: "src/ops/vulkan/is_finite/bin",
        workgroup_size: [256, 1, 1],
        push_constant_size: std::mem::size_of::<IsFinitePush>() as u32,
    };
    let dispatched = dispatch_with_standard_bindings(
        runtime,
        &spec,
        desc_bytes.as_slice(),
        io_buffers.input_bytes.as_slice(),
        io_buffers.output_bytes.as_mut_slice(),
        io_buffers.output_alias_input,
        io_buffers.output_offset,
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
        kind: OpKind::IsFinite,
        mode,
        broadcast: false,
        in0: inputs[0].dtype(),
        in1: None,
        out0: output_dtype,
    };
    let kernel = crate::ops::cpu::registry::lookup_kernel(key)?;
    kernel(attrs, inputs, output)
}
