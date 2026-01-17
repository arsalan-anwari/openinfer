#[cfg(feature = "vulkan")]
use std::collections::HashMap;
#[cfg(feature = "vulkan")]
use std::sync::Arc;

#[cfg(feature = "vulkan")]
use serde_json::Value;

#[cfg(feature = "vulkan")]
use crate::graph::OpKind;
use crate::tensor::{DType, TensorValue};

#[cfg(feature = "vulkan")]
pub mod vulkan;
pub mod cpu;

#[cfg(feature = "vulkan")]
#[derive(Debug, Clone)]
pub struct OpShaderInfo {
    pub settings: HashMap<String, Value>,
    pub spv_by_target: HashMap<String, &'static [u8]>,
}

#[cfg(feature = "vulkan")]
pub trait ShaderRegistry {
    fn shader_for_op(&self, op: OpKind) -> Option<Arc<OpShaderInfo>>;
}

#[cfg(feature = "vulkan")]
#[derive(Debug, Clone)]
pub struct VulkanBuffer {
    pub dtype: DType,
    pub len: usize,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub shader: Option<Arc<OpShaderInfo>>,
    pub inner: Arc<crate::backend::vulkan::VulkanBufferInner>,
}

#[cfg(feature = "vulkan")]
#[derive(Debug, Clone)]
pub enum DeviceTensor {
    Vulkan(VulkanBuffer),
}

#[derive(Debug, Clone)]
pub enum TensorStorage {
    Host(TensorValue),
    #[cfg(feature = "vulkan")]
    Device(DeviceTensor),
}

// TensorStorage is moved across threads but not shared concurrently.
unsafe impl Send for TensorStorage {}

impl TensorStorage {
    pub fn dtype(&self) -> DType {
        match self {
            TensorStorage::Host(value) => value.dtype(),
            #[cfg(feature = "vulkan")]
            TensorStorage::Device(DeviceTensor::Vulkan(buf)) => buf.dtype,
        }
    }
}

#[cfg(feature = "vulkan")]
impl VulkanBuffer {
    pub fn with_shader(mut self, shader: Option<Arc<OpShaderInfo>>) -> Self {
        self.shader = shader;
        self
    }

    pub fn shader_info(&self) -> Option<&OpShaderInfo> {
        self.shader.as_deref()
    }

    pub fn shader_setting_bool(&self, key: &str) -> Option<bool> {
        self.shader_info()?
            .settings
            .get(key)
            .and_then(|value| value.as_bool())
    }

    pub fn spv_bytes_for_target(&self, target: &str) -> Option<&'static [u8]> {
        self.shader_info()?.spv_by_target.get(target).copied()
    }
}
