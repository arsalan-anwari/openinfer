use anyhow::{anyhow, Result};

use crate::graph::{OpAttrs, OpKind};
use crate::op_defs::{op_schema, TypeRule};
use crate::simulator::Device;
use crate::tensor::{DType, TensorValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpMode {
    Normal,
    Inplace,
    Accumulate,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpKey {
    pub kind: OpKind,
    pub mode: OpMode,
    pub broadcast: bool,
    pub inputs: Vec<DType>,
    pub out0: DType,
}

pub type KernelFn = fn(&OpAttrs, &[TensorValue], Option<&mut TensorValue>) -> Result<()>;

#[allow(unused)]
pub fn op_supports_dtype(kind: OpKind, mode: OpMode, in0: DType, out0: DType) -> bool {
    let schema = match op_schema(kind) {
        Some(schema) => schema,
        None => return false,
    };
    let support = match schema.dtype_support {
        Some(support) => support,
        None => return true,
    };
    match mode {
        OpMode::Accumulate => support
            .accumulate
            .iter()
            .any(|(in_dtype, out_dtype)| *in_dtype == in0 && *out_dtype == out0),
        OpMode::Normal | OpMode::Inplace => support.normal.contains(&in0),
    }
}

pub fn build_op_entries_same_input(
    kind: OpKind,
    kernel_for_mode: impl Fn(OpMode) -> Option<KernelFn>,
) -> Result<Vec<(OpKey, KernelFn)>> {
    let schema = op_schema(kind).ok_or_else(|| anyhow!("missing op schema {:?}", kind))?;
    let support = schema
        .dtype_support
        .ok_or_else(|| anyhow!("op {:?} has no dtype support", kind))?;
    let inputs = schema.inputs.fixed().ok_or_else(|| {
        anyhow!(
            "op {:?} has non-fixed input arity {:?}",
            kind,
            schema.inputs
        )
    })?;
    let broadcast_flags: &[bool] = if schema.broadcast.allow() {
        &[false, true]
    } else {
        &[false]
    };

    let mut entries = Vec::new();
    for in_dtype in support.normal {
        let out_dtype = match schema.type_rule {
            TypeRule::SameAsInput(0) => *in_dtype,
            TypeRule::Fixed(dtype) => dtype,
            _ => {
                return Err(anyhow!(
                    "op {:?} has unsupported type rule for entry build",
                    kind
                ))
            }
        };
        for &broadcast in broadcast_flags {
            let normal_key = OpKey {
                kind,
                mode: OpMode::Normal,
                broadcast,
                inputs: vec![*in_dtype; inputs],
                out0: out_dtype,
            };
            if let Some(kernel) = kernel_for_mode(OpMode::Normal) {
                entries.push((normal_key, kernel));
            }
            if schema.inplace.allow() {
                let inplace_key = OpKey {
                    kind,
                    mode: OpMode::Inplace,
                    broadcast,
                    inputs: vec![*in_dtype; inputs],
                    out0: out_dtype,
                };
                if let Some(kernel) = kernel_for_mode(OpMode::Inplace) {
                    entries.push((inplace_key, kernel));
                }
            }
        }
    }
    if schema.accumulate.allow() {
        for (in_dtype, out_dtype) in support.accumulate {
            for &broadcast in broadcast_flags {
                let acc_key = OpKey {
                    kind,
                    mode: OpMode::Accumulate,
                    broadcast,
                    inputs: vec![*in_dtype; inputs],
                    out0: *out_dtype,
                };
                if let Some(kernel) = kernel_for_mode(OpMode::Accumulate) {
                    entries.push((acc_key, kernel));
                }
            }
        }
    }
    Ok(entries)
}

#[allow(unused)]
pub fn build_op_entries_with_outputs(
    kind: OpKind,
    output_dtypes: &[DType],
    kernel_for_mode: impl Fn(OpMode) -> Option<KernelFn>,
) -> Result<Vec<(OpKey, KernelFn)>> {
    let schema = op_schema(kind).ok_or_else(|| anyhow!("missing op schema {:?}", kind))?;
    let support = schema
        .dtype_support
        .ok_or_else(|| anyhow!("op {:?} has no dtype support", kind))?;
    let inputs = schema.inputs.fixed().ok_or_else(|| {
        anyhow!(
            "op {:?} has non-fixed input arity {:?}",
            kind,
            schema.inputs
        )
    })?;
    let broadcast_flags: &[bool] = if schema.broadcast.allow() {
        &[false, true]
    } else {
        &[false]
    };

    let mut entries = Vec::new();
    for in_dtype in support.normal {
        for &out_dtype in output_dtypes {
            for &broadcast in broadcast_flags {
                let normal_key = OpKey {
                    kind,
                    mode: OpMode::Normal,
                    broadcast,
                    inputs: vec![*in_dtype; inputs],
                    out0: out_dtype,
                };
                if let Some(kernel) = kernel_for_mode(OpMode::Normal) {
                    entries.push((normal_key, kernel));
                }
                if schema.inplace.allow() {
                    let inplace_key = OpKey {
                        kind,
                        mode: OpMode::Inplace,
                        broadcast,
                        inputs: vec![*in_dtype; inputs],
                        out0: out_dtype,
                    };
                    if let Some(kernel) = kernel_for_mode(OpMode::Inplace) {
                        entries.push((inplace_key, kernel));
                    }
                }
            }
        }
    }
    if schema.accumulate.allow() {
        for (in_dtype, out_dtype) in support.accumulate {
            for &broadcast in broadcast_flags {
                let acc_key = OpKey {
                    kind,
                    mode: OpMode::Accumulate,
                    broadcast,
                    inputs: vec![*in_dtype; inputs],
                    out0: *out_dtype,
                };
                if let Some(kernel) = kernel_for_mode(OpMode::Accumulate) {
                    entries.push((acc_key, kernel));
                }
            }
        }
    }
    Ok(entries)
}

pub fn lookup_kernel(device: Device, key: OpKey) -> Result<KernelFn> {
    match device {
        Device::Cpu => crate::ops::cpu::registry::lookup_kernel(key),
        Device::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                crate::ops::vulkan::registry::lookup_kernel(key)
            }
            #[cfg(not(feature = "vulkan"))]
            {
                Err(anyhow!("device {:?} requires the vulkan feature", device))
            }
        }
    }
}
