use anyhow::{anyhow, Result};

use crate::backend::{DeviceTensor, TensorStorage, VulkanBuffer};
use crate::graph::{OpAttrs, OpKind};
use crate::ops::{lookup_kernel, KernelFn};
use crate::tensor::{DType, TensorValue};

use super::{Device, DeviceBackend};

#[derive(Debug)]
pub struct VulkanBackend;

impl VulkanBackend {
    pub fn new() -> Self {
        Self
    }
}

impl DeviceBackend for VulkanBackend {
    fn device(&self) -> Device {
        Device::Vulkan
    }

    fn alloc(&self, dtype: DType, len: usize) -> Result<TensorStorage> {
        println!("vulkan alloc: dtype={:?} len={}", dtype, len);
        Ok(TensorStorage::Device(DeviceTensor::Vulkan(VulkanBuffer {
            dtype,
            len,
        })))
    }

    fn upload(&self, value: TensorValue) -> Result<TensorStorage> {
        let dtype = value.dtype();
        let len = value.len();
        println!("vulkan upload: dtype={:?} len={}", dtype, len);
        Ok(TensorStorage::Device(DeviceTensor::Vulkan(VulkanBuffer {
            dtype,
            len,
        })))
    }

    fn download(&self, value: TensorStorage) -> Result<TensorValue> {
        match value {
            TensorStorage::Host(_) => Err(anyhow!("vulkan backend cannot download host tensor")),
            TensorStorage::Device(DeviceTensor::Vulkan(buf)) => {
                println!("vulkan download: dtype={:?} len={}", buf.dtype, buf.len);
                Ok(TensorValue::zeros(buf.dtype, buf.len))
            }
        }
    }

    fn exec_op(
        &self,
        op: OpKind,
        attrs: &OpAttrs,
        dtype: DType,
        tensors: &[TensorStorage],
    ) -> Result<TensorStorage> {
        let buffers = to_vulkan_buffers(tensors)?;
        let kernel = lookup_kernel(self.device(), op, dtype, *attrs)
            .ok_or_else(|| anyhow!("unsupported op {}", op.as_str()))?;
        match kernel {
            KernelFn::Vulkan(func) => Ok(TensorStorage::Device(DeviceTensor::Vulkan(
                func(attrs, &buffers)?,
            ))),
            KernelFn::Host(_) => Err(anyhow!("vulkan backend cannot run host kernel")),
        }
    }
}

fn to_vulkan_buffers<'a>(tensors: &'a [TensorStorage]) -> Result<Vec<&'a VulkanBuffer>> {
    let mut out = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        match tensor {
            TensorStorage::Device(DeviceTensor::Vulkan(buf)) => out.push(buf),
            TensorStorage::Host(_) => return Err(anyhow!("host tensor passed to vulkan backend")),
        }
    }
    Ok(out)
}
