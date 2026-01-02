use anyhow::{anyhow, Result};

use crate::backend::{DeviceTensor, TensorStorage};
use crate::graph::{OpAttrs, OpKind};
use crate::ops::{lookup_kernel, KernelFn};
use crate::tensor::{DType, TensorValue};

use super::{Device, DeviceBackend};

#[derive(Debug)]
pub struct CpuBackend {
    device: Device,
}

impl CpuBackend {
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

impl DeviceBackend for CpuBackend {
    fn device(&self) -> Device {
        self.device
    }

    fn alloc(&self, dtype: DType, len: usize) -> Result<TensorStorage> {
        Ok(TensorStorage::Host(TensorValue::zeros(dtype, len)))
    }

    fn upload(&self, value: TensorValue) -> Result<TensorStorage> {
        Ok(TensorStorage::Host(value))
    }

    fn download(&self, value: TensorStorage) -> Result<TensorValue> {
        match value {
            TensorStorage::Host(host) => Ok(host),
            TensorStorage::Device(_) => Err(anyhow!("host backend cannot download device tensor")),
        }
    }

    fn exec_op(
        &self,
        op: OpKind,
        attrs: &OpAttrs,
        output_dtype: DType,
        tensors: &[TensorStorage],
    ) -> Result<TensorStorage> {
        let input_dtypes: Vec<DType> = tensors.iter().map(|t| t.dtype()).collect();
        let host = to_host_tensors(tensors)?;
        let kernel = lookup_kernel(self.device, op, output_dtype, &input_dtypes, *attrs)
            .ok_or_else(|| anyhow!("unsupported op {}", op.as_str()))?;
        match kernel {
            KernelFn::Host(func) => Ok(TensorStorage::Host((func)(attrs, &host)?)),
            #[cfg(feature = "vulkan")]
            KernelFn::Vulkan(_) => Err(anyhow!("host backend cannot run device kernel")),
        }
    }
}

fn to_host_tensors(tensors: &[TensorStorage]) -> Result<Vec<TensorValue>> {
    let mut out = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        match tensor {
            TensorStorage::Host(value) => out.push(value.clone()),
            TensorStorage::Device(DeviceTensor::Vulkan(_)) => {
                return Err(anyhow!("device tensor passed to host backend"));
            }
        }
    }
    Ok(out)
}
