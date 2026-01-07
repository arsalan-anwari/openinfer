use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::graph::OpKind;
use crate::tensor::{DType, TensorValue};

#[cfg(feature = "vulkan")]
pub mod vulkan;
pub mod cpu;

#[cfg(not(feature = "vulkan"))]
pub mod vulkan {
    #[derive(Debug)]
    pub struct VulkanBufferInner;

    #[derive(Debug)]
    #[allow(unused)]
    pub struct VulkanRuntime;
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OpShaderInfo {
    pub path: String,
    pub spv_dir: String,
    pub push_constants_size: usize,
    pub settings: HashMap<String, Value>,
    pub spv_by_target: HashMap<String, &'static [u8]>,
}

#[allow(unused)]
pub trait ShaderRegistry {
    fn shader_for_op(&self, op: OpKind) -> Option<Arc<OpShaderInfo>>;
}

#[derive(Debug, Clone)]
pub struct VulkanBuffer {
    pub dtype: DType,
    pub len: usize,
    pub shader: Option<Arc<OpShaderInfo>>,
    #[allow(unused)]
    pub inner: Arc<crate::backend::vulkan::VulkanBufferInner>,
}

#[derive(Debug, Clone)]
#[allow(unused)]
pub enum DeviceTensor {
    Vulkan(VulkanBuffer),
}

#[derive(Debug, Clone)]
pub enum TensorStorage {
    Host(TensorValue),
    #[allow(unused)]
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

#[allow(unused)]
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
