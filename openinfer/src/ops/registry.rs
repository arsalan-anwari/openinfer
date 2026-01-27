use anyhow::{anyhow, Result};

use crate::graph::{OpAttrs, OpKind};
use crate::registry::op_schema;
use crate::simulator::Device;
use crate::tensor::{DType, TensorValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpMode {
    Normal,
    Inplace,
    Accumulate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpKey {
    pub kind: OpKind,
    pub mode: OpMode,
    pub broadcast: bool,
    pub in0: DType,
    pub in1: Option<DType>,
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

pub fn lookup_kernel(device: Device, key: OpKey) -> Result<KernelFn> {
    match device {
        Device::Cpu => crate::ops::cpu::registry::lookup_kernel(key),
        Device::CpuAvx | Device::CpuAvx2 => {
            Err(anyhow!("device {:?} registry not implemented", device))
        }
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
