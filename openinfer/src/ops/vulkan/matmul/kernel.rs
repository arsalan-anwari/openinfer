use anyhow::{anyhow, Result};

use crate::graph::{OpAttrs, OpKind};
use crate::ops::cpu::broadcast::broadcast_shape;
use crate::ops::registry::{op_supports_dtype, OpKey, OpMode};
use crate::registry::acc_dtype;
use crate::tensor::{DType, TensorValue};

use bytemuck::{Pod, Zeroable};

use crate::ops::vulkan::descriptor::{build_tensor_desc, MAX_DIMS};
use crate::ops::vulkan::dispatch::VulkanOpSpec;
use crate::ops::vulkan::op_helpers::{
    build_desc_bytes, build_output_desc, dispatch_with_standard_bindings,
    prepare_matmul_staging_io, return_staging_buffers, target_name,
};
use crate::ops::vulkan::runtime::get_vulkan_runtime;
use crate::ops::vulkan::tensor_bytes::write_tensor_from_bytes;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct MatmulPush {
    len: u32,
    tensor_count: u32,
    params_offset: u32,
    flags: u32,
}

pub fn matmul_normal_dispatch(
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    dispatch_matmul(OpMode::Normal, attrs, inputs, output)
}

pub fn matmul_inplace_dispatch(
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    dispatch_matmul(OpMode::Inplace, attrs, inputs, output)
}

pub fn matmul_accumulate_dispatch(
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    dispatch_matmul(OpMode::Accumulate, attrs, inputs, output)
}

fn dispatch_matmul(
    mode: OpMode,
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    crate::vk_trace!(
        "dispatch_matmul mode={:?} input_dtype={:?} output_dtype={:?}",
        mode,
        inputs.get(0).map(|t| t.dtype()),
        output.as_ref().map(|t| t.dtype())
    );
    if inputs.len() != 2 {
        return Err(anyhow!("matmul expects 2 inputs, got {}", inputs.len()));
    }
    let output = output.ok_or_else(|| anyhow!("matmul requires an output tensor"))?;
    let input_dtype = inputs[0].dtype();
    let output_dtype = match mode {
        OpMode::Accumulate => acc_dtype(attrs)?,
        _ => input_dtype,
    };

    let a_shape = inputs[0].shape();
    let b_shape = inputs[1].shape();
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(anyhow!("matmul expects inputs with rank >= 2"));
    }
    let a_rank = a_shape.len();
    let b_rank = b_shape.len();
    let a_m = a_shape[a_rank - 2];
    let a_k = a_shape[a_rank - 1];
    let b_k = b_shape[b_rank - 2];
    let b_n = b_shape[b_rank - 1];
    if a_k != b_k {
        return Err(anyhow!(
            "matmul inner dim mismatch: {} vs {}",
            a_k,
            b_k
        ));
    }
    let a_batch = &a_shape[..a_rank - 2];
    let b_batch = &b_shape[..b_rank - 2];
    let batch_shape = broadcast_shape(a_batch, b_batch)?;
    let mut expected = batch_shape.clone();
    expected.push(a_m);
    expected.push(b_n);
    if output.shape() != expected.as_slice() {
        return Err(anyhow!(
            "output shape {:?} does not match matmul shape {:?}",
            output.shape(),
            expected
        ));
    }
    let out_rank = expected.len();
    if out_rank > MAX_DIMS {
        return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype, a_batch != b_batch);
    }

    let runtime = match get_vulkan_runtime() {
        Some(runtime) => runtime,
        None => {
            crate::vk_trace!("vulkan runtime not initialized, falling back to cpu");
            return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype, a_batch != b_batch);
        }
    };

    if !runtime.caps().supports_dtype(input_dtype) || !runtime.caps().supports_dtype(output_dtype) {
        crate::vk_trace!(
            "dtype unsupported by vulkan caps (input={:?}, output={:?}), cpu fallback",
            input_dtype,
            output_dtype
        );
        return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype, a_batch != b_batch);
    }
    if !op_supports_dtype(OpKind::Matmul, mode, input_dtype, output_dtype) {
        crate::vk_trace!(
            "vulkan target unsupported (mode={:?}, in={:?}, out={:?}), cpu fallback",
            mode,
            input_dtype,
            output_dtype
        );
        return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype, a_batch != b_batch);
    }

    let mut io_buffers =
        prepare_matmul_staging_io(mode, inputs, output, output_dtype, false)?;

    let input0_is_output = mode == OpMode::Inplace;
    let input0 = if input0_is_output { &*output } else { &inputs[0] };
    let mut descs = Vec::with_capacity(3);
    descs.push(build_tensor_desc(input0, out_rank, io_buffers.input0_offset)?);
    descs.push(build_tensor_desc(&inputs[1], out_rank, io_buffers.input1_offset)?);
    descs.push(build_output_desc(output, out_rank, 0)?);

    let push = MatmulPush {
        len: output.len() as u32,
        tensor_count: descs.len() as u32,
        params_offset: 0,
        flags: 0,
    };

    let target = match target_name(OpKind::Matmul, mode, input_dtype, output_dtype) {
        Ok(name) => name,
        Err(_) => {
            crate::vk_trace!("target name not found, cpu fallback");
            return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype, a_batch != b_batch);
        }
    };
    crate::vk_trace!("dispatching vulkan matmul target={}", target);
    let desc_bytes = build_desc_bytes(&descs);
    let push_bytes = bytemuck::bytes_of(&push).to_vec();
    let spec = VulkanOpSpec {
        entry: &target,
        spv_dir: "src/ops/vulkan/matmul/bin",
        workgroup_size: [256, 1, 1],
        push_constant_size: std::mem::size_of::<MatmulPush>() as u32,
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

    cpu_fallback(mode, attrs, inputs, Some(output), output_dtype, a_batch != b_batch)
}

fn cpu_fallback(
    mode: OpMode,
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
    output_dtype: DType,
    broadcast: bool,
) -> Result<()> {
    let key = OpKey {
        kind: OpKind::Matmul,
        mode,
        broadcast,
        inputs: inputs.iter().map(|tensor| tensor.dtype()).collect(),
        out0: output_dtype,
    };
    let kernel = crate::ops::cpu::registry::lookup_kernel(key)?;
    kernel(attrs, inputs, output)
}
