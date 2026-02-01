use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs, OpKind};
use crate::ops::registry::{op_supports_dtype, OpKey, OpMode};
use crate::tensor::{DType, TensorValue};
use crate::types::dtype_suffix;

use bytemuck::{Pod, Zeroable};

use crate::ops::vulkan::descriptor::{build_tensor_desc, MAX_DIMS};
use crate::ops::vulkan::dispatch::VulkanOpSpec;
use crate::ops::vulkan::op_helpers::{
    build_desc_bytes, build_output_desc, dispatch_with_standard_bindings, prepare_unary_staging_io,
    return_staging_buffers,
};
use crate::ops::vulkan::runtime::get_vulkan_runtime;
use crate::ops::vulkan::tensor_bytes::write_tensor_from_bytes;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct CastPush {
    len: u32,
    tensor_count: u32,
    params_offset: u32,
    flags: u32,
}

pub fn cast_normal_dispatch(
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    dispatch_cast(OpMode::Normal, attrs, inputs, output)
}

fn dispatch_cast(
    mode: OpMode,
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    crate::vk_trace!(
        "dispatch_cast mode={:?} input_dtype={:?} output_dtype={:?}",
        mode,
        inputs.get(0).map(|t| t.dtype()),
        output.as_ref().map(|t| t.dtype())
    );
    if inputs.len() != 1 {
        return Err(anyhow!("cast expects 1 input, got {}", inputs.len()));
    }
    let output = output.ok_or_else(|| anyhow!("cast requires an output tensor"))?;
    let input_dtype = inputs[0].dtype();
    let output_dtype = get_to_dtype(attrs)?;
    if output.dtype() != output_dtype {
        return Err(anyhow!(
            "cast output dtype {:?} does not match attr {:?}",
            output.dtype(),
            output_dtype
        ));
    }
    if !is_allowed_cast(input_dtype, output_dtype) {
        return Err(anyhow!(
            "unsupported cast from {:?} to {:?}",
            input_dtype,
            output_dtype
        ));
    }

    let (out_rank, exceeds_rank) =
        crate::ops::vulkan::op_helpers::validate_unary_shape_and_rank(
            &inputs[0],
            output,
            MAX_DIMS,
        )?;
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
    if !op_supports_dtype(OpKind::Cast, mode, input_dtype, output_dtype) {
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

    let mut descs = Vec::with_capacity(2);
    descs.push(build_tensor_desc(&inputs[0], out_rank, io_buffers.input_offset)?);
    descs.push(build_output_desc(output, out_rank, 0)?);

    let push = CastPush {
        len: output.len() as u32,
        tensor_count: descs.len() as u32,
        params_offset: 0,
        flags: 0,
    };

    let in_name = dtype_suffix(input_dtype)?;
    let entry = if input_dtype.is_packed() {
        format!("cast_{in_name}_packed")
    } else {
        format!("cast_{in_name}_normal")
    };

    crate::vk_trace!("dispatching vulkan cast target={}", entry);
    let desc_bytes = build_desc_bytes(&descs);
    let push_bytes = bytemuck::bytes_of(&push).to_vec();
    let spec = VulkanOpSpec {
        entry: &entry,
        spv_dir: "src/ops/vulkan/cast/bin",
        workgroup_size: [256, 1, 1],
        push_constant_size: std::mem::size_of::<CastPush>() as u32,
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

fn get_to_dtype(attrs: &OpAttrs) -> Result<DType> {
    attrs
        .items
        .iter()
        .find(|attr| attr.name == "to")
        .ok_or_else(|| anyhow!("missing to attribute"))
        .and_then(|attr| match &attr.value {
            AttrValue::DType(dtype) => Ok(*dtype),
            _ => Err(anyhow!("to attribute must be a dtype")),
        })
}

fn is_allowed_cast(in_dtype: DType, out_dtype: DType) -> bool {
    if is_float(out_dtype) {
        return is_packed_signed(in_dtype)
            || is_packed_unsigned(in_dtype)
            || is_signed_int(in_dtype)
            || is_unsigned_int(in_dtype)
            || is_float(in_dtype);
    }
    if is_signed_int(out_dtype) {
        if is_float(in_dtype) {
            return true;
        }
        if is_packed_signed(in_dtype) {
            return out_dtype.bit_width() > in_dtype.bit_width();
        }
        if is_signed_int(in_dtype) {
            return out_dtype.bit_width() > in_dtype.bit_width();
        }
        return false;
    }
    if is_unsigned_int(out_dtype) {
        if is_float(in_dtype) {
            return true;
        }
        if is_packed_unsigned(in_dtype) {
            return out_dtype.bit_width() > in_dtype.bit_width();
        }
        if is_unsigned_int(in_dtype) {
            return out_dtype.bit_width() > in_dtype.bit_width();
        }
        return false;
    }
    false
}

fn is_float(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F8 | DType::F16 | DType::BF16 | DType::F32 | DType::F64
    )
}

fn is_signed_int(dtype: DType) -> bool {
    matches!(dtype, DType::I8 | DType::I16 | DType::I32 | DType::I64)
}

fn is_unsigned_int(dtype: DType) -> bool {
    matches!(dtype, DType::U8 | DType::U16 | DType::U32 | DType::U64)
}

fn is_packed_signed(dtype: DType) -> bool {
    matches!(dtype, DType::I1 | DType::I2 | DType::I4)
}

fn is_packed_unsigned(dtype: DType) -> bool {
    matches!(dtype, DType::U1 | DType::U2 | DType::U4)
}

fn cpu_fallback(
    mode: OpMode,
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
    output_dtype: DType,
) -> Result<()> {
    let key = OpKey {
        kind: OpKind::Cast,
        mode,
        broadcast: false,
        inputs: inputs.iter().map(|tensor| tensor.dtype()).collect(),
        out0: output_dtype,
    };
    let kernel = crate::ops::cpu::registry::lookup_kernel(key)?;
    kernel(attrs, inputs, output)
}
