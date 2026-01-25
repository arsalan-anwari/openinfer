use anyhow::{anyhow, Result};

#[cfg(feature = "vulkan")]
use crate::backend::DeviceTensor;
use crate::backend::TensorStorage;
use crate::graph::{OpAttrs, OpKind};
use crate::ops::{lookup_kernel, KernelFn};
use crate::ops::registry::lookup_kernel_inplace;
use crate::simulator::{Device, DeviceBackend};
use crate::tensor::{DType, TensorValue};
use crate::types::cpu::{effective_dtype, from_effective_tensor, to_effective_tensor};

pub mod scheduler;

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

    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<TensorStorage> {
        Ok(TensorStorage::Host(TensorValue::zeros(dtype, shape)))
    }

    fn upload(&self, value: TensorValue) -> Result<TensorStorage> {
        Ok(TensorStorage::Host(value))
    }

    fn download(&self, value: TensorStorage) -> Result<TensorValue> {
        match value {
            TensorStorage::Host(host) => Ok(host),
            #[cfg(feature = "vulkan")]
            TensorStorage::Device(_) => Err(anyhow!("host backend cannot download device tensor")),
        }
    }

    fn exec_op(
        &self,
        op: OpKind,
        attrs: &OpAttrs,
        output_dtype: DType,
        tensors: &[TensorStorage],
        output: Option<TensorStorage>,
        thread_id: usize,
    ) -> Result<TensorStorage> {
        let output_effective = effective_dtype(output_dtype)?;
        let mut host = to_host_tensors(tensors)?;
        host = host
            .into_iter()
            .map(|value| {
                let effective = effective_dtype(value.dtype())?;
                to_effective_tensor(value, effective)
            })
            .collect::<Result<Vec<_>>>()?;
        let input_dtypes: Vec<DType> = host.iter().map(|t| t.dtype()).collect();
        let mut lookup_device = self.device;
        if matches!(self.device, Device::CpuAvx | Device::CpuAvx2) {
            let needs_broadcast_fallback = match op {
                OpKind::Add | OpKind::Mul => host
                    .iter()
                    .skip(1)
                    .any(|value| value.shape() != host[0].shape()),
                OpKind::Matmul => {
                    if host.len() >= 2 {
                        host[0].shape() != host[1].shape()
                    } else {
                        false
                    }
                }
                _ => false,
            };
            if needs_broadcast_fallback {
                lookup_device = Device::Cpu;
            }
        }
        let kernel = lookup_kernel(lookup_device, op, output_effective, &input_dtypes, attrs)
            .ok_or_else(|| anyhow!("unsupported op {}", op.as_str()))?;
        match kernel {
            KernelFn::Host(func) => {
                let mut output_value = match output {
                    Some(TensorStorage::Host(value)) => {
                        Some(to_effective_tensor(value, output_effective)?)
                    }
                    _ => None,
                };
                let output_value_result = (func)(attrs, &host, output_value.as_mut(), thread_id)?;
                if let Some(value) = output_value_result {
                    let value = from_effective_tensor(value, output_dtype)?;
                    return Ok(TensorStorage::Host(value));
                }
                if let Some(existing) = output_value {
                    let value = from_effective_tensor(existing, output_dtype)?;
                    return Ok(TensorStorage::Host(value));
                }
                Err(anyhow!("cpu kernel returned no output"))
            }
            #[cfg(feature = "vulkan")]
            KernelFn::Vulkan(_) => Err(anyhow!("host backend cannot run device kernel")),
        }
    }

    fn exec_op_inplace(
        &self,
        op: OpKind,
        attrs: &OpAttrs,
        output_dtype: DType,
        tensors: &[TensorStorage],
        thread_id: usize,
    ) -> Result<TensorStorage> {
        let output_effective = effective_dtype(output_dtype)?;
        let mut host = to_host_tensors(tensors)?;
        host = host
            .into_iter()
            .map(|value| {
                let effective = effective_dtype(value.dtype())?;
                to_effective_tensor(value, effective)
            })
            .collect::<Result<Vec<_>>>()?;
        if host.is_empty() {
            return Err(anyhow!("inplace op {} expects at least 1 input", op.as_str()));
        }
        let mut output = host.remove(0);
        let input_dtypes: Vec<DType> = std::iter::once(&output)
            .chain(host.iter())
            .map(|t| t.dtype())
            .collect();
        let mut lookup_device = self.device;
        if matches!(self.device, Device::CpuAvx | Device::CpuAvx2) {
            let needs_broadcast_fallback = match op {
                OpKind::Add | OpKind::Mul => host
                    .iter()
                    .any(|value| value.shape() != output.shape()),
                OpKind::Matmul => {
                    if host.len() >= 1 {
                        output.shape() != host[0].shape()
                    } else {
                        false
                    }
                }
                _ => false,
            };
            if needs_broadcast_fallback {
                lookup_device = Device::Cpu;
            }
        }
        let kernel = lookup_kernel_inplace(lookup_device, op, output_effective, &input_dtypes, attrs)
            .ok_or_else(|| anyhow!("unsupported inplace op {}", op.as_str()))?;
        match kernel {
            crate::ops::registry::InplaceKernelFn::Host(func) => {
                (func)(attrs, &mut output, &host, thread_id)?;
                let output = from_effective_tensor(output, output_dtype)?;
                Ok(TensorStorage::Host(output))
            }
            #[cfg(feature = "vulkan")]
            crate::ops::registry::InplaceKernelFn::Vulkan(_) => {
                Err(anyhow!("host backend cannot run device kernel"))
            }
        }
    }
}

fn to_host_tensors(tensors: &[TensorStorage]) -> Result<Vec<TensorValue>> {
    let mut out = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        match tensor {
            TensorStorage::Host(value) => out.push(value.clone()),
            #[cfg(feature = "vulkan")]
            TensorStorage::Device(DeviceTensor::Vulkan(_)) => {
                return Err(anyhow!("device tensor passed to host backend"));
            }
        }
    }
    Ok(out)
}
