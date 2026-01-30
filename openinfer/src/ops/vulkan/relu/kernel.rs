use anyhow::{anyhow, Result};

use crate::graph::{OpAttrs, OpKind};
use crate::ops::registry::{op_supports_dtype, OpKey, OpMode};
use crate::tensor::{DType, TensorValue};

use bytemuck::{Pod, Zeroable};

use crate::ops::vulkan::common::collect_scalar_attr_bits;
use crate::ops::vulkan::descriptor::{build_tensor_desc, MAX_DIMS};
use crate::ops::vulkan::dispatch::VulkanOpSpec;
use crate::ops::vulkan::op_helpers::{
    build_desc_bytes, build_output_desc, dispatch_with_standard_bindings, prepare_unary_staging_io,
    return_staging_buffers, target_name, validate_unary_shape_and_rank,
};
use crate::ops::vulkan::runtime::get_vulkan_runtime;
use crate::ops::vulkan::tensor_bytes::write_tensor_from_bytes;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ReluPush {
    len: u32,
    tensor_count: u32,
    params_offset: u32,
    flags: u32,
    alpha_f32: f32,
    clamp_max_f32: f32,
    alpha_i32: i32,
    clamp_max_i32: i32,
    alpha_i64_lo: u32,
    alpha_i64_hi: u32,
    clamp_i64_lo: u32,
    clamp_i64_hi: u32,
    alpha_f64_lo: u32,
    alpha_f64_hi: u32,
    clamp_f64_lo: u32,
    clamp_f64_hi: u32,
}

pub fn relu_normal_dispatch(
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    dispatch_relu(OpMode::Normal, attrs, inputs, output)
}

pub fn relu_inplace_dispatch(
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    dispatch_relu(OpMode::Inplace, attrs, inputs, output)
}

fn dispatch_relu(
    mode: OpMode,
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    crate::vk_trace!(
        "dispatch_relu mode={:?} input_dtype={:?} output_dtype={:?}",
        mode,
        inputs.get(0).map(|t| t.dtype()),
        output.as_ref().map(|t| t.dtype())
    );
    if inputs.len() != 1 {
        return Err(anyhow!("relu expects 1 input, got {}", inputs.len()));
    }
    let output = output.ok_or_else(|| anyhow!("relu requires an output tensor"))?;
    let input_dtype = inputs[0].dtype();
    let output_dtype = input_dtype;

    let (out_rank, exceeds_rank) =
        validate_unary_shape_and_rank(&inputs[0], output, MAX_DIMS)?;
    if exceeds_rank {
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
    if !op_supports_dtype(OpKind::Relu, mode, input_dtype, output_dtype) {
        crate::vk_trace!(
            "vulkan target unsupported (mode={:?}, in={:?}, out={:?}), cpu fallback",
            mode,
            input_dtype,
            output_dtype
        );
        return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype);
    }

    let scalar_attrs = collect_scalar_attr_bits(OpKind::Relu, input_dtype, attrs)?;
    let alpha_bits = scalar_attrs.get("alpha").copied();
    let clamp_bits = scalar_attrs.get("clamp_max").copied();
    let mut alpha_f64 = 0.0;
    let mut clamp_max_f64 = f64::INFINITY;
    let mut alpha_i64 = 0i64;
    let mut clamp_max_i64 = i64::MAX;
    match input_dtype {
        DType::F8 | DType::BF16 | DType::F16 | DType::F32 | DType::F64 => {
            if let Some(bits) = alpha_bits {
                alpha_f64 = bits.as_f64();
            }
            if let Some(bits) = clamp_bits {
                clamp_max_f64 = bits.as_f64();
            }
        }
        DType::I4 | DType::I8 | DType::I16 | DType::I32 | DType::I64 => {
            if let Some(bits) = alpha_bits {
                alpha_i64 = bits.as_i64();
            }
            if let Some(bits) = clamp_bits {
                clamp_max_i64 = bits.as_i64();
            }
        }
        _ => {}
    }
    let alpha_f32 = alpha_f64 as f32;
    let clamp_max_f32 = clamp_max_f64 as f32;
    let alpha_i32 = alpha_i64 as i32;
    let clamp_max_i32 = clamp_max_i64 as i32;
    let alpha_i64_bits = alpha_i64 as u64;
    let clamp_i64_bits = clamp_max_i64 as u64;
    let alpha_f64_bits = alpha_f64.to_bits();
    let clamp_f64_bits = clamp_max_f64.to_bits();

    let mut io_buffers =
        prepare_unary_staging_io(mode, &inputs[0], output, output_dtype)?;

    let mut descs = Vec::with_capacity(if mode == OpMode::Inplace { 1 } else { 2 });
    if mode == OpMode::Inplace {
        descs.push(build_tensor_desc(output, out_rank, io_buffers.input_offset)?);
    } else {
        descs.push(build_tensor_desc(&inputs[0], out_rank, io_buffers.input_offset)?);
        descs.push(build_output_desc(output, out_rank, 0)?);
    }

    let push = ReluPush {
        len: output.len() as u32,
        tensor_count: descs.len() as u32,
        params_offset: 0,
        flags: 0,
        alpha_f32,
        clamp_max_f32,
        alpha_i32,
        clamp_max_i32,
        alpha_i64_lo: alpha_i64_bits as u32,
        alpha_i64_hi: (alpha_i64_bits >> 32) as u32,
        clamp_i64_lo: clamp_i64_bits as u32,
        clamp_i64_hi: (clamp_i64_bits >> 32) as u32,
        alpha_f64_lo: alpha_f64_bits as u32,
        alpha_f64_hi: (alpha_f64_bits >> 32) as u32,
        clamp_f64_lo: clamp_f64_bits as u32,
        clamp_f64_hi: (clamp_f64_bits >> 32) as u32,
    };

    let target = match target_name(OpKind::Relu, mode, input_dtype, output_dtype) {
        Ok(name) => name,
        Err(_) => {
            crate::vk_trace!("target name not found, cpu fallback");
            return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype);
        }
    };
    crate::vk_trace!("dispatching vulkan relu target={}", target);
    let desc_bytes = build_desc_bytes(&descs);
    let push_bytes = bytemuck::bytes_of(&push).to_vec();
    let spec = VulkanOpSpec {
        entry: &target,
        spv_dir: "src/ops/vulkan/relu/bin",
        workgroup_size: [256, 1, 1],
        push_constant_size: std::mem::size_of::<ReluPush>() as u32,
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
        kind: OpKind::Relu,
        mode,
        broadcast: false,
        in0: inputs[0].dtype(),
        in1: None,
        out0: output_dtype,
    };
    let kernel = crate::ops::cpu::registry::lookup_kernel(key)?;
    kernel(attrs, inputs, output)
}
