use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs, OpKind};
use crate::ops::cpu::broadcast::broadcast_shape;
use crate::ops::registry::{OpKey, OpMode};
use crate::tensor::{DType, TensorValue};

use bytemuck::{Pod, Zeroable};

use crate::ops::vulkan::descriptor::{build_tensor_desc, dtype_code, TensorDesc, MAX_DIMS};
use crate::ops::vulkan::runtime::get_vulkan_runtime;
use crate::ops::vulkan::staging_buffer::{return_staging, take_staging, StagingBuffers};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct PushConstants {
    len: u32,
    tensor_count: u32,
    params_offset: u32,
    flags: u32,
}

pub fn dispatch_add_normal(
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    dispatch_add(OpMode::Normal, attrs, inputs, output)
}

pub fn dispatch_add_inplace(
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    dispatch_add(OpMode::Inplace, attrs, inputs, output)
}

pub fn dispatch_add_accumulate(
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    dispatch_add(OpMode::Accumulate, attrs, inputs, output)
}

fn dispatch_add(
    mode: OpMode,
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    crate::vk_trace!(
        "dispatch_add mode={:?} input_dtype={:?} output_dtype={:?}",
        mode,
        inputs.get(0).map(|t| t.dtype()),
        output.as_ref().map(|t| t.dtype())
    );
    if inputs.len() != 2 {
        return Err(anyhow!("add expects 2 inputs, got {}", inputs.len()));
    }
    let output = output.ok_or_else(|| anyhow!("add requires an output tensor"))?;
    let input_dtype = inputs[0].dtype();
    let output_dtype = match mode {
        OpMode::Accumulate => acc_dtype(attrs)?,
        _ => input_dtype,
    };

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
    if out_shape.len() > MAX_DIMS {
        return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype, broadcast);
    }

    let _key = OpKey {
        kind: OpKind::Add,
        mode,
        broadcast,
        in0: input_dtype,
        in1: Some(input_dtype),
        out0: output_dtype,
    };

    let runtime = match get_vulkan_runtime() {
        Some(runtime) => runtime,
        None => {
            crate::vk_trace!("vulkan runtime not initialized, falling back to cpu");
            return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype, broadcast);
        }
    };

    if !runtime.caps().supports_dtype(input_dtype) || !runtime.caps().supports_dtype(output_dtype) {
        crate::vk_trace!(
            "dtype unsupported by vulkan caps (input={:?}, output={:?}), cpu fallback",
            input_dtype,
            output_dtype
        );
        return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype, broadcast);
    }
    if !vulkan_target_supported(mode, input_dtype, output_dtype) {
        crate::vk_trace!(
            "vulkan target unsupported (mode={:?}, in={:?}, out={:?}), cpu fallback",
            mode,
            input_dtype,
            output_dtype
        );
        return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype, broadcast);
    }

    let input0_source = if mode == OpMode::Inplace { &*output } else { &inputs[0] };
    let output_len = tensor_byte_len(output.dtype(), output.len());
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

    let mut descs = Vec::with_capacity(if mode == OpMode::Inplace { 2 } else { 3 });
    if mode == OpMode::Inplace {
        descs.push(build_tensor_desc(output, out_shape.len(), input0_offset)?);
        descs.push(build_tensor_desc(&inputs[1], out_shape.len(), input1_offset)?);
    } else {
        descs.push(build_tensor_desc(&inputs[0], out_shape.len(), input0_offset)?);
        descs.push(build_tensor_desc(&inputs[1], out_shape.len(), input1_offset)?);
        descs.push(build_output_desc(output, out_shape.len(), 0)?);
    }

    let push = PushConstants {
        len: output.len() as u32,
        tensor_count: descs.len() as u32,
        params_offset: 0,
        flags: 0,
    };

    let target = match target_name(mode, input_dtype, output_dtype, runtime.caps().float16) {
        Ok(name) => name,
        Err(_) => {
            crate::vk_trace!("target name not found, cpu fallback");
            return cpu_fallback(mode, attrs, inputs, Some(output), output_dtype, broadcast);
        }
    };
    crate::vk_trace!("dispatching vulkan add target={}", target);
    let desc_bytes = bytemuck::cast_slice(&descs).to_vec();
    let push_bytes = bytemuck::bytes_of(&push).to_vec();
    let output_alias_input = mode == OpMode::Inplace;
    let output_offset = input0_offset as u64;
    let mut dispatched = runtime.dispatch_add(
        &target,
        &desc_bytes,
        &input_bytes,
        output_bytes.as_mut_slice(),
        &push_bytes,
        output_offset,
        output_alias_input,
    );
    if let Err(err) = &dispatched {
        crate::vk_trace!("vulkan dispatch error: {}", err);
    }
    if dispatched.is_err() && input_dtype == DType::F16 {
        let fallback_target = target_name(mode, input_dtype, output_dtype, false)?;
        crate::vk_trace!(
            "native f16 dispatch failed, retrying simulated target={}",
            fallback_target
        );
        dispatched = runtime.dispatch_add(
            &fallback_target,
            &desc_bytes,
            &input_bytes,
            output_bytes.as_mut_slice(),
            &push_bytes,
            output_offset,
            output_alias_input,
        );
        if let Err(err) = &dispatched {
            crate::vk_trace!("vulkan dispatch error (simulated f16): {}", err);
        }
    }
    if dispatched.is_ok() {
        write_tensor_from_bytes(output, &output_bytes)?;
        crate::vk_trace!("vulkan dispatch successful");
        return_staging(StagingBuffers {
            input: input_bytes,
            output: output_bytes,
        })?;
        return Ok(());
    }
    crate::vk_trace!("vulkan dispatch failed, cpu fallback");
    return_staging(StagingBuffers {
        input: input_bytes,
        output: output_bytes,
    })?;

    cpu_fallback(mode, attrs, inputs, Some(output), output_dtype, broadcast)
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
        kind: OpKind::Add,
        mode,
        broadcast,
        in0: inputs[0].dtype(),
        in1: Some(inputs[0].dtype()),
        out0: output_dtype,
    };
    let kernel = crate::ops::cpu::registry::lookup_kernel(key)?;
    kernel(attrs, inputs, output)
}

fn build_output_desc(value: &TensorValue, out_rank: usize, byte_offset: u32) -> Result<TensorDesc> {
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

fn target_name(mode: OpMode, in_dtype: DType, out_dtype: DType, use_native_f16: bool) -> Result<String> {
    let in_name = dtype_suffix(in_dtype)?;
    let out_name = dtype_suffix(out_dtype)?;
    if in_dtype.is_packed() {
        return Ok(match mode {
            OpMode::Normal => format!("add_{}_packed", in_name),
            OpMode::Inplace => format!("add_{}_packed_inplace", in_name),
            OpMode::Accumulate => format!("add_{}_accumulate_{}", in_name, out_name),
        });
    }
    if in_dtype == DType::F16 {
        let suffix = if use_native_f16 { "native" } else { "simulated" };
        return Ok(match mode {
            OpMode::Normal => format!("add_{}_normal_{}", in_name, suffix),
            OpMode::Inplace => format!("add_{}_inplace_{}", in_name, suffix),
            OpMode::Accumulate => format!("add_{}_accumulate_{}_{}", in_name, out_name, suffix),
        });
    }
    Ok(match mode {
        OpMode::Normal => format!("add_{}_normal", in_name),
        OpMode::Inplace => format!("add_{}_inplace", in_name),
        OpMode::Accumulate => format!("add_{}_accumulate_{}", in_name, out_name),
    })
}

fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F8E5M2 => Ok("f8"),
        DType::BF16 => Ok("bf16"),
        DType::F16 => Ok("f16"),
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        DType::I1 => Ok("i1"),
        DType::I2 => Ok("i2"),
        DType::I4 => Ok("i4"),
        DType::I8 => Ok("i8"),
        DType::I16 => Ok("i16"),
        DType::I32 => Ok("i32"),
        DType::I64 => Ok("i64"),
        DType::U1 => Ok("u1"),
        DType::U2 => Ok("u2"),
        DType::U4 => Ok("u4"),
        DType::U8 => Ok("u8"),
        DType::U16 => Ok("u16"),
        DType::U32 => Ok("u32"),
        DType::U64 => Ok("u64"),
        DType::Bool => Ok("bool"),
        DType::Bitset => Ok("bitset"),
        DType::T1 | DType::T2 => Err(anyhow!("packed ternary types not supported in vulkan")),
    }
}

fn vulkan_target_supported(mode: OpMode, in_dtype: DType, out_dtype: DType) -> bool {
    match mode {
        OpMode::Normal | OpMode::Inplace => matches!(
            in_dtype,
            DType::F32
                | DType::F16
                | DType::BF16
                | DType::F8E5M2
                | DType::F64
                | DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
                | DType::Bool
                | DType::Bitset
                | DType::I1
                | DType::I2
                | DType::I4
                | DType::U1
                | DType::U2
                | DType::U4
        ),
        OpMode::Accumulate => matches!(
            (in_dtype, out_dtype),
            (DType::I1, DType::I64)
                | (DType::I2, DType::I64)
                | (DType::I4, DType::I64)
                | (DType::I8, DType::I64)
                | (DType::I16, DType::I64)
                | (DType::I32, DType::I64)
                | (DType::U1, DType::U64)
                | (DType::U2, DType::U64)
                | (DType::U4, DType::U64)
                | (DType::U8, DType::U64)
                | (DType::U16, DType::U64)
                | (DType::U32, DType::U64)
        ),
    }
}

#[allow(dead_code)]
fn tensor_to_bytes(value: &TensorValue) -> Result<Vec<u8>> {
    Ok(match value {
        TensorValue::F32(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::F64(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::I8(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::I16(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::I32(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::I64(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::U8(tensor) => tensor.data.clone(),
        TensorValue::U16(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::U32(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::U64(tensor) => bytemuck::cast_slice(&tensor.data).to_vec(),
        TensorValue::Bool(tensor) => tensor.data.iter().map(|v| if *v { 1 } else { 0 }).collect(),
        TensorValue::Bitset(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::F16(tensor) => tensor.data.iter().flat_map(|v| v.bits.to_le_bytes()).collect(),
        TensorValue::BF16(tensor) => tensor.data.iter().flat_map(|v| v.bits.to_le_bytes()).collect(),
        TensorValue::F8E5M2(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::I4(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::I2(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::I1(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::U4(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::U2(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::U1(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::T1(_) | TensorValue::T2(_) => {
            return Err(anyhow!("ternary packed types not supported in vulkan"))
        }
    })
}

fn tensor_append_bytes(value: &TensorValue, out: &mut Vec<u8>) -> Result<()> {
    match value {
        TensorValue::F32(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::F64(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::I8(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::I16(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::I32(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::I64(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::U8(tensor) => out.extend_from_slice(&tensor.data),
        TensorValue::U16(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::U32(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::U64(tensor) => out.extend_from_slice(bytemuck::cast_slice(&tensor.data)),
        TensorValue::Bool(tensor) => out.extend(tensor.data.iter().map(|v| if *v { 1 } else { 0 })),
        TensorValue::Bitset(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::F16(tensor) => out.extend(tensor.data.iter().flat_map(|v| v.bits.to_le_bytes())),
        TensorValue::BF16(tensor) => out.extend(tensor.data.iter().flat_map(|v| v.bits.to_le_bytes())),
        TensorValue::F8E5M2(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::I4(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::I2(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::I1(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::U4(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::U2(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::U1(tensor) => out.extend(tensor.data.iter().map(|v| v.bits)),
        TensorValue::T1(_) | TensorValue::T2(_) => {
            return Err(anyhow!("ternary packed types not supported in vulkan"))
        }
    }
    Ok(())
}


fn write_tensor_from_bytes(output: &mut TensorValue, bytes: &[u8]) -> Result<()> {
    match output {
        TensorValue::F32(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::F64(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::I8(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::I16(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::I32(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::I64(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::U8(tensor) => {
            tensor.data = bytes.to_vec();
        }
        TensorValue::U16(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::U32(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::U64(tensor) => {
            tensor.data = bytemuck::cast_slice(bytes).to_vec();
        }
        TensorValue::Bool(tensor) => {
            tensor.data = bytes.iter().map(|v| *v != 0).collect();
        }
        TensorValue::Bitset(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::Bitset { bits: *v }).collect();
        }
        TensorValue::F16(tensor) => {
            let mut out = Vec::new();
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(crate::tensor::F16 { bits });
            }
            tensor.data = out;
        }
        TensorValue::BF16(tensor) => {
            let mut out = Vec::new();
            for chunk in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(crate::tensor::BF16 { bits });
            }
            tensor.data = out;
        }
        TensorValue::F8E5M2(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::F8E5M2 { bits: *v }).collect();
        }
        TensorValue::I4(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::I4 { bits: *v }).collect();
        }
        TensorValue::I2(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::I2 { bits: *v }).collect();
        }
        TensorValue::I1(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::I1 { bits: *v }).collect();
        }
        TensorValue::U4(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::U4 { bits: *v }).collect();
        }
        TensorValue::U2(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::U2 { bits: *v }).collect();
        }
        TensorValue::U1(tensor) => {
            tensor.data = bytes.iter().map(|v| crate::tensor::U1 { bits: *v }).collect();
        }
        TensorValue::T1(_) | TensorValue::T2(_) => {
            return Err(anyhow!("ternary packed types not supported in vulkan"))
        }
    }
    Ok(())
}

fn tensor_byte_len(dtype: DType, logical_len: usize) -> usize {
    if dtype.is_packed() {
        dtype.storage_len(logical_len)
    } else {
        let elem_bytes = (dtype.bit_width() as usize + 7) / 8;
        logical_len.saturating_mul(elem_bytes)
    }
}

fn acc_dtype(attrs: &OpAttrs) -> Result<DType> {
    attrs
        .items
        .iter()
        .find(|attr| attr.name == "acc")
        .ok_or_else(|| anyhow!("missing acc attribute"))
        .and_then(|attr| match &attr.value {
            AttrValue::DType(dtype) => Ok(*dtype),
            _ => Err(anyhow!("acc attribute must be a dtype")),
        })
}
