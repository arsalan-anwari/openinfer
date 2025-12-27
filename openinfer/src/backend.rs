use crate::tensor::{DType, TensorValue};

#[derive(Debug, Clone)]
pub struct VulkanBuffer {
    pub dtype: DType,
    pub len: usize,
}

#[derive(Debug, Clone)]
pub enum DeviceTensor {
    Vulkan(VulkanBuffer),
}

#[derive(Debug, Clone)]
pub enum TensorStorage {
    Host(TensorValue),
    Device(DeviceTensor),
}

impl TensorStorage {
    pub fn dtype(&self) -> DType {
        match self {
            TensorStorage::Host(value) => value.dtype(),
            TensorStorage::Device(DeviceTensor::Vulkan(buf)) => buf.dtype,
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        match self {
            TensorStorage::Host(value) => value.len(),
            TensorStorage::Device(DeviceTensor::Vulkan(buf)) => buf.len,
        }
    }
}
