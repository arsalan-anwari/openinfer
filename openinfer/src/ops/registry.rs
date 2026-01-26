use anyhow::{anyhow, Result};

use crate::graph::{OpAttrs, OpKind};
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

pub fn lookup_kernel(device: Device, key: OpKey) -> Result<KernelFn> {
    match device {
        Device::Cpu => crate::ops::cpu::registry::lookup_kernel(key),
        Device::CpuAvx | Device::CpuAvx2 => {
            Err(anyhow!("device {:?} registry not implemented", device))
        }
    }
}
