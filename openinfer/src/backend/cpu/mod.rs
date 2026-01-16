use anyhow::{anyhow, Result};

#[cfg(feature = "vulkan")]
use crate::backend::DeviceTensor;
use crate::backend::TensorStorage;
use crate::graph::{OpAttrs, OpKind};
use crate::ops::{broadcast_enabled, lookup_kernel, KernelFn};
use crate::ops::registry::lookup_kernel_inplace;
use crate::simulator::{Device, DeviceBackend};
use crate::tensor::{broadcast_shapes, DType, TensorValue};

pub mod broadcast;

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
        thread_id: usize,
    ) -> Result<TensorStorage> {
        let input_dtypes: Vec<DType> = tensors.iter().map(|t| t.dtype()).collect();
        let mut host = to_host_tensors(tensors)?;
        if host.len() > 1 {
            if broadcast_enabled(op, self.device) {
                let mut out_shape = host[0].shape().to_vec();
                for value in host.iter().skip(1) {
                    out_shape = broadcast_shapes(&out_shape, value.shape())?;
                }
                host = host
                    .iter()
                    .map(|value| broadcast::broadcast_value_to_shape(value, &out_shape))
                    .collect::<Result<Vec<_>>>()?;
            } else {
                let first = host[0].shape();
                for value in host.iter().skip(1) {
                    if value.shape() != first {
                        return Err(anyhow!("op {} requires identical input shapes on {:?}", op.as_str(), self.device));
                    }
                }
            }
        }
        let kernel = lookup_kernel(self.device, op, output_dtype, &input_dtypes, attrs)
            .ok_or_else(|| anyhow!("unsupported op {}", op.as_str()))?;
        match kernel {
            KernelFn::Host(func) => Ok(TensorStorage::Host((func)(attrs, &host, thread_id)?)),
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
        let input_dtypes: Vec<DType> = tensors.iter().map(|t| t.dtype()).collect();
        let mut host = to_host_tensors(tensors)?;
        if host.is_empty() {
            return Err(anyhow!("inplace op {} expects at least 1 input", op.as_str()));
        }
        let mut output = host.remove(0);
        let kernel = lookup_kernel_inplace(self.device, op, output_dtype, &input_dtypes, attrs)
            .ok_or_else(|| anyhow!("unsupported inplace op {}", op.as_str()))?;
        match kernel {
            crate::ops::registry::InplaceKernelFn::Host(func) => {
                (func)(attrs, &mut output, &host, thread_id)?;
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
