use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::backend::{DeviceTensor, OpShaderInfo, ShaderRegistry, TensorStorage, VulkanBuffer};
use crate::graph::{OpAttrs, OpKind};
use crate::ops::{lookup_kernel, KernelFn};
use crate::simulator::{Device, DeviceBackend};
use crate::tensor::{Bitset, DType, F16, TensorValue};

pub mod runtime;
pub use runtime::{storage_size_bytes, VulkanBufferInner, VulkanRuntime};

mod embedded_spirv {
    include!(concat!(env!("OUT_DIR"), "/vulkan_spirv.rs"));
}

#[derive(Debug, Default)]
pub struct VulkanShaderRegistry {
    ops: HashMap<String, Arc<OpShaderInfo>>,
}

#[derive(Debug, Deserialize)]
struct VulkanShaderManifest {
    ops: HashMap<String, VulkanShaderEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct VulkanShaderEntry {
    path: String,
    spv_dir: String,
    push_constants_size: usize,
    #[serde(default)]
    settings: HashMap<String, Value>,
}

impl VulkanShaderRegistry {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read vulkan shader manifest at {}", path.display()))?;
        let manifest: VulkanShaderManifest =
            serde_json::from_str(&contents).with_context(|| {
                format!("failed to parse vulkan shader manifest at {}", path.display())
            })?;
        let mut ops = HashMap::new();
        for (name, entry) in manifest.ops {
            if !matches!(name.as_str(), "abs" | "add" | "mul" | "relu") {
                continue;
            }
            let spv_by_target = embedded_spirv::embedded_spirv_for_op(name.as_str());
            ops.insert(
                name,
                Arc::new(OpShaderInfo {
                    path: entry.path,
                    spv_dir: entry.spv_dir,
                    push_constants_size: entry.push_constants_size,
                    settings: entry.settings,
                    spv_by_target,
                }),
            );
        }
        Ok(Self { ops })
    }

    pub fn load_default() -> Result<Self> {
        Self::load_from_file(Self::default_manifest_path())
    }

    pub fn default_manifest_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/ops/vulkan/shaders.json")
    }
}

impl ShaderRegistry for VulkanShaderRegistry {
    fn shader_for_op(&self, op: OpKind) -> Option<Arc<OpShaderInfo>> {
        self.ops.get(op.as_str()).cloned()
    }
}

#[derive(Debug)]
pub struct VulkanBackend {
    shaders: VulkanShaderRegistry,
    runtime: Mutex<Option<Arc<VulkanRuntime>>>,
}

impl VulkanBackend {
    pub fn new() -> Self {
        let shaders = VulkanShaderRegistry::load_default().unwrap_or_else(|err| {
            eprintln!("vulkan shaders: {}", err);
            VulkanShaderRegistry::default()
        });
        Self {
            shaders,
            runtime: Mutex::new(None),
        }
    }

    fn runtime(&self) -> Result<Arc<VulkanRuntime>> {
        let mut guard = self
            .runtime
            .lock()
            .map_err(|_| anyhow!("vulkan runtime lock poisoned"))?;
        if let Some(runtime) = guard.as_ref() {
            return Ok(Arc::clone(runtime));
        }
        let runtime = Arc::new(VulkanRuntime::new()?);
        *guard = Some(Arc::clone(&runtime));
        Ok(runtime)
    }

    fn ensure_supported_dtype(&self, dtype: DType) -> Result<()> {
        let runtime = self.runtime()?;
        match dtype {
            DType::I64 | DType::U64 => {
                if !runtime.supports_i64() {
                    return Err(anyhow!("vulkan device does not support i64/u64"));
                }
            }
            DType::F32
            | DType::I8
            | DType::I16
            | DType::I32
            | DType::U8
            | DType::U16
            | DType::U32
            | DType::Bool => {}
            _ => {
                return Err(anyhow!("vulkan backend does not support dtype {:?}", dtype));
            }
        }
        Ok(())
    }
}

impl DeviceBackend for VulkanBackend {
    fn device(&self) -> Device {
        Device::Vulkan
    }

    fn alloc(&self, dtype: DType, len: usize) -> Result<TensorStorage> {
        let runtime = self.runtime()?;
        self.ensure_supported_dtype(dtype)?;
        let size = storage_size_bytes(dtype) * len;
        let inner = runtime.create_buffer(size)?;
        Ok(TensorStorage::Device(DeviceTensor::Vulkan(VulkanBuffer {
            dtype,
            len,
            shader: None,
            inner,
        })))
    }

    fn upload(&self, value: TensorValue) -> Result<TensorStorage> {
        let runtime = self.runtime()?;
        let dtype = value.dtype();
        self.ensure_supported_dtype(dtype)?;
        let len = value.len();
        let size = storage_size_bytes(dtype) * len;
        let inner = runtime.create_buffer(size)?;
        let bytes = encode_tensor(value);
        runtime.write_buffer(&inner, &bytes)?;
        Ok(TensorStorage::Device(DeviceTensor::Vulkan(VulkanBuffer {
            dtype,
            len,
            shader: None,
            inner,
        })))
    }

    fn download(&self, value: TensorStorage) -> Result<TensorValue> {
        match value {
            TensorStorage::Host(_) => Err(anyhow!("vulkan backend cannot download host tensor")),
            TensorStorage::Device(DeviceTensor::Vulkan(buf)) => {
                let runtime = self.runtime()?;
                let size = storage_size_bytes(buf.dtype) * buf.len;
                let mut bytes = vec![0u8; size];
                runtime.read_buffer(&buf.inner, &mut bytes)?;
                decode_tensor(buf.dtype, buf.len, &bytes)
            }
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
        let shader = self.shaders.shader_for_op(op);
        if shader.is_none() {
            eprintln!("vulkan shaders: missing entry for op {}", op.as_str());
        }
        let buffers = to_vulkan_buffers(tensors, shader.clone())?;
        let buffer_refs: Vec<&VulkanBuffer> = buffers.iter().collect();
        let kernel = lookup_kernel(self.device(), op, output_dtype, &input_dtypes, attrs)
            .ok_or_else(|| anyhow!("unsupported op {}", op.as_str()))?;
        match kernel {
            KernelFn::Vulkan(func) => Ok(TensorStorage::Device(DeviceTensor::Vulkan(
                (func)(attrs, &buffer_refs, thread_id)?,
            ))),
            KernelFn::Host(_) => Err(anyhow!("vulkan backend cannot run host kernel")),
        }
    }
}

fn to_vulkan_buffers(
    tensors: &[TensorStorage],
    shader: Option<Arc<OpShaderInfo>>,
) -> Result<Vec<VulkanBuffer>> {
    let mut out = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        match tensor {
            TensorStorage::Device(DeviceTensor::Vulkan(buf)) => {
                out.push(buf.clone().with_shader(shader.clone()));
            }
            TensorStorage::Host(_) => return Err(anyhow!("host tensor passed to vulkan backend")),
        }
    }
    Ok(out)
}

fn encode_tensor(value: TensorValue) -> Vec<u8> {
    match value {
        TensorValue::I8(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| (i32::from(*v)).to_le_bytes())
            .collect(),
        TensorValue::I16(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| (i32::from(*v)).to_le_bytes())
            .collect(),
        TensorValue::I32(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        TensorValue::I64(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        TensorValue::U8(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| u32::from(*v).to_le_bytes())
            .collect(),
        TensorValue::U16(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| u32::from(*v).to_le_bytes())
            .collect(),
        TensorValue::U32(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        TensorValue::U64(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        TensorValue::F16(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| u32::from(v.bits).to_le_bytes())
            .collect(),
        TensorValue::F32(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        TensorValue::F64(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        TensorValue::Bool(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| u32::from(*v as u8).to_le_bytes())
            .collect(),
        TensorValue::Bitset(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| u32::from(v.bits).to_le_bytes())
            .collect(),
    }
}

fn decode_tensor(dtype: DType, len: usize, bytes: &[u8]) -> Result<TensorValue> {
    match dtype {
        DType::I8 => Ok(TensorValue::I8(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as i8)
                .collect(),
        ))),
        DType::I16 => Ok(TensorValue::I16(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as i16)
                .collect(),
        ))),
        DType::I32 => Ok(TensorValue::I32(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        DType::I64 => Ok(TensorValue::I64(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(8)
                .take(len)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        DType::U8 => Ok(TensorValue::U8(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()) as u8)
                .collect(),
        ))),
        DType::U16 => Ok(TensorValue::U16(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()) as u16)
                .collect(),
        ))),
        DType::U32 => Ok(TensorValue::U32(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        DType::U64 => Ok(TensorValue::U64(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(8)
                .take(len)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        DType::F16 => Ok(TensorValue::F16(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| F16 {
                    bits: u32::from_le_bytes(chunk.try_into().unwrap()) as u16,
                })
                .collect(),
        ))),
        DType::F32 => Ok(TensorValue::F32(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        DType::F64 => Ok(TensorValue::F64(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(8)
                .take(len)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        DType::Bool => Ok(TensorValue::Bool(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()) != 0)
                .collect(),
        ))),
        DType::Bitset => Ok(TensorValue::Bitset(crate::tensor::Tensor::new(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| Bitset {
                    bits: u32::from_le_bytes(chunk.try_into().unwrap()) as u8,
                })
                .collect(),
        ))),
    }
}
