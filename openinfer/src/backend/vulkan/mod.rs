use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::backend::{DeviceTensor, OpShaderInfo, ShaderRegistry, TensorStorage, VulkanBuffer};
use crate::graph::{OpAttrs, OpKind};
use crate::ops::{broadcast_enabled, lookup_kernel, KernelFn};
use crate::ops::registry::lookup_kernel_inplace;
use crate::simulator::{Device, DeviceBackend};
use crate::tensor::{broadcast_shapes, BF16, Bitset, DType, F16, F8E5M2, I1, I2, I4, TensorValue};
use crate::types::vulkan::{from_effective_tensor, to_effective_tensor};

pub mod runtime;
pub use runtime::{storage_size_bytes_for_len, VulkanBufferInner, VulkanRuntime};
pub mod broadcast;
pub mod scheduler;

mod embedded_spirv {
    include!(concat!(env!("OUT_DIR"), "/vulkan_spirv.rs"));
}

pub(crate) fn embedded_spirv_for_op(op: &str) -> HashMap<String, &'static [u8]> {
    embedded_spirv::embedded_spirv_for_op(op)
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
            let spv_by_target = embedded_spirv::embedded_spirv_for_op(name.as_str());
            ops.insert(
                name,
                Arc::new(OpShaderInfo {
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

    pub fn shader_for_name(&self, name: &str) -> Option<Arc<OpShaderInfo>> {
        self.ops.get(name).cloned()
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
    dispatch_lock: Mutex<()>,
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
            dispatch_lock: Mutex::new(()),
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

    fn effective_dtype(&self, dtype: DType) -> Result<DType> {
        let runtime = self.runtime()?;
        match dtype {
            DType::F16 | DType::BF16 | DType::F8E5M2 => Ok(DType::F32),
            DType::I64 | DType::U64 => {
                if !runtime.supports_i64() {
                    return Err(anyhow!("vulkan device does not support i64/u64 (shader_int64)"));
                }
                Ok(dtype)
            }
            DType::F64 => {
                if !runtime.supports_f64() {
                    return Err(anyhow!("vulkan device does not support f64 (shader_float64)"));
                }
                Ok(dtype)
            }
            DType::F32
            | DType::I8
            | DType::I16
            | DType::I32
            | DType::U8
            | DType::U16
            | DType::U32
            | DType::Bool => Ok(dtype),
            DType::Bitset => Err(anyhow!("vulkan backend does not support bitset tensors")),
            _ => Ok(dtype)
        }
    }
}

impl DeviceBackend for VulkanBackend {
    fn device(&self) -> Device {
        Device::Vulkan
    }

    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<TensorStorage> {
        let _guard = self
            .dispatch_lock
            .lock()
            .map_err(|_| anyhow!("vulkan dispatch lock poisoned"))?;
        let runtime = self.runtime()?;
        let effective_dtype = self.effective_dtype(dtype)?;
        let len = crate::tensor::numel(shape);
        let size = storage_size_bytes_for_len(effective_dtype, len);
        let inner = runtime.create_buffer(size)?;
        let strides = crate::tensor::compute_strides(shape);
        Ok(TensorStorage::Device(DeviceTensor::Vulkan(VulkanBuffer {
            dtype,
            effective_dtype,
            len,
            shape: shape.to_vec(),
            strides,
            shader: None,
            inner,
        })))
    }

    fn upload(&self, value: TensorValue) -> Result<TensorStorage> {
        let _guard = self
            .dispatch_lock
            .lock()
            .map_err(|_| anyhow!("vulkan dispatch lock poisoned"))?;
        let runtime = self.runtime()?;
        let dtype = value.dtype();
        let effective_dtype = self.effective_dtype(dtype)?;
        let len = value.len();
        let shape = value.shape().to_vec();
        let strides = value.strides().to_vec();
        let size = storage_size_bytes_for_len(effective_dtype, len);
        let inner = runtime.create_buffer(size)?;
        let encoded = to_effective_tensor(value, effective_dtype)?;
        let bytes = encode_tensor(encoded);
        runtime.write_buffer(&inner, &bytes)?;
        Ok(TensorStorage::Device(DeviceTensor::Vulkan(VulkanBuffer {
            dtype,
            effective_dtype,
            len,
            shape,
            strides,
            shader: None,
            inner,
        })))
    }

    fn download(&self, value: TensorStorage) -> Result<TensorValue> {
        let _guard = self
            .dispatch_lock
            .lock()
            .map_err(|_| anyhow!("vulkan dispatch lock poisoned"))?;
        match value {
            TensorStorage::Host(_) => Err(anyhow!("vulkan backend cannot download host tensor")),
            TensorStorage::Device(DeviceTensor::Vulkan(buf)) => {
                let runtime = self.runtime()?;
                let size = storage_size_bytes_for_len(buf.effective_dtype, buf.len);
                let mut bytes = vec![0u8; size];
                runtime.read_buffer(&buf.inner, &mut bytes)?;
                let value = decode_tensor(buf.effective_dtype, buf.shape.as_slice(), &bytes)?;
                from_effective_tensor(value, buf.dtype)
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
        let output_effective = self.effective_dtype(output_dtype)?;
        let shader = self.shaders.shader_for_op(op);
        if shader.is_none() {
            eprintln!("vulkan shaders: missing entry for op {}", op.as_str());
        }
        let _guard = self
            .dispatch_lock
            .lock()
            .map_err(|_| anyhow!("vulkan dispatch lock poisoned"))?;
        let mut buffers = to_vulkan_buffers(tensors, shader.clone())?;
        if buffers.len() > 1 {
            if broadcast_enabled(op, self.device()) {
                let mut out_shape = buffers[0].shape.clone();
                for buffer in buffers.iter().skip(1) {
                    out_shape = broadcast_shapes(&out_shape, buffer.shape.as_slice())?;
                }
                buffers = buffers
                    .into_iter()
                    .map(|buffer| {
                        if buffer.shape == out_shape {
                            Ok(buffer)
                        } else {
                            crate::backend::vulkan::broadcast::broadcast_buffer(
                                &buffer,
                                out_shape.as_slice(),
                                thread_id,
                            )
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;
            }
        }
        let input_dtypes: Vec<DType> = buffers.iter().map(|b| b.effective_dtype).collect();
        let buffer_refs: Vec<&VulkanBuffer> = buffers.iter().collect();
        let kernel = lookup_kernel(self.device(), op, output_effective, &input_dtypes, attrs)
            .ok_or_else(|| anyhow!("unsupported op {}", op.as_str()))?;
        match kernel {
            KernelFn::Vulkan(func) => Ok(TensorStorage::Device(DeviceTensor::Vulkan(
                (func)(attrs, &buffer_refs, thread_id)?,
            ))),
            KernelFn::Host(_) => Err(anyhow!("vulkan backend cannot run host kernel")),
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
        let output_effective = self.effective_dtype(output_dtype)?;
        let shader = self
            .shaders
            .shader_for_name(&format!("{}_inplace", op.as_str()));
        if shader.is_none() {
            eprintln!(
                "vulkan shaders: missing entry for op {}_inplace",
                op.as_str()
            );
        }
        let _guard = self
            .dispatch_lock
            .lock()
            .map_err(|_| anyhow!("vulkan dispatch lock poisoned"))?;
        let buffers = to_vulkan_buffers(tensors, shader)?;
        if buffers.len() > 1 {
            let first = buffers[0].shape.clone();
            for buffer in buffers.iter().skip(1) {
                if buffer.shape != first {
                    return Err(anyhow!(
                        "op {} inplace requires identical input shapes on {:?}",
                        op.as_str(),
                        self.device()
                    ));
                }
            }
        }
        let input_dtypes: Vec<DType> = buffers.iter().map(|b| b.effective_dtype).collect();
        let buffer_refs: Vec<&VulkanBuffer> = buffers.iter().collect();
        let kernel = lookup_kernel_inplace(self.device(), op, output_effective, &input_dtypes, attrs)
            .ok_or_else(|| anyhow!("unsupported inplace op {}", op.as_str()))?;
        match kernel {
            crate::ops::registry::InplaceKernelFn::Vulkan(func) => {
                Ok(TensorStorage::Device(DeviceTensor::Vulkan(
                (func)(attrs, &buffer_refs, thread_id)?,
                )))
            }
            crate::ops::registry::InplaceKernelFn::Host(_) => {
                Err(anyhow!("vulkan backend cannot run host kernel"))
            }
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
        TensorValue::BF16(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| u32::from(v.bits).to_le_bytes())
            .collect(),
        TensorValue::F8E5M2(tensor) => tensor
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
        TensorValue::I4(tensor) => pack_packed_bits(tensor.data.iter().map(|v| v.bits), 4, tensor.len()),
        TensorValue::I2(tensor) => pack_packed_bits(tensor.data.iter().map(|v| v.bits), 2, tensor.len()),
        TensorValue::I1(tensor) => pack_packed_bits(tensor.data.iter().map(|v| v.bits), 1, tensor.len()),
    }
}

fn decode_tensor(dtype: DType, shape: &[usize], bytes: &[u8]) -> Result<TensorValue> {
    let len = crate::tensor::numel(shape);
    match dtype {
        DType::I8 => Ok(TensorValue::I8(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as i8)
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::I16 => Ok(TensorValue::I16(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as i16)
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::I32 => Ok(TensorValue::I32(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::I64 => Ok(TensorValue::I64(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(8)
                .take(len)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::U8 => Ok(TensorValue::U8(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()) as u8)
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::U16 => Ok(TensorValue::U16(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()) as u16)
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::U32 => Ok(TensorValue::U32(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::U64 => Ok(TensorValue::U64(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(8)
                .take(len)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::F16 => Ok(TensorValue::F16(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| F16 {
                    bits: u32::from_le_bytes(chunk.try_into().unwrap()) as u16,
                })
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::BF16 => Ok(TensorValue::BF16(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| BF16 {
                    bits: u32::from_le_bytes(chunk.try_into().unwrap()) as u16,
                })
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::F8E5M2 => Ok(TensorValue::F8E5M2(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| F8E5M2 {
                    bits: u32::from_le_bytes(chunk.try_into().unwrap()) as u8,
                })
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::F32 => Ok(TensorValue::F32(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::F64 => Ok(TensorValue::F64(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(8)
                .take(len)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::Bool => Ok(TensorValue::Bool(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()) != 0)
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::Bitset => Ok(TensorValue::Bitset(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| Bitset {
                    bits: u32::from_le_bytes(chunk.try_into().unwrap()) as u8,
                })
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::I4 => {
            let values = unpack_packed_bits(bytes, 4, len)?;
            Ok(TensorValue::I4(crate::tensor::Tensor::from_vec_with_opts(
                values.into_iter().map(|bits| I4 { bits }).collect(),
                crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                },
            )?))
        }
        DType::I2 => {
            let values = unpack_packed_bits(bytes, 2, len)?;
            Ok(TensorValue::I2(crate::tensor::Tensor::from_vec_with_opts(
                values.into_iter().map(|bits| I2 { bits }).collect(),
                crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                },
            )?))
        }
        DType::I1 => {
            let values = unpack_packed_bits(bytes, 1, len)?;
            Ok(TensorValue::I1(crate::tensor::Tensor::from_vec_with_opts(
                values.into_iter().map(|bits| I1 { bits }).collect(),
                crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    ..crate::tensor::TensorOptions::default()
                },
            )?))
        }
    }
}

fn pack_packed_bits<I: Iterator<Item = u8>>(values: I, bits_per: u8, len: usize) -> Vec<u8> {
    let total_bits = len.saturating_mul(bits_per as usize);
    let total_bytes = (total_bits + 7) / 8;
    let mut out = vec![0u8; total_bytes];
    let mask = (1u8 << bits_per) - 1;
    for (idx, value) in values.take(len).enumerate() {
        let bit_index = idx * bits_per as usize;
        let byte_index = bit_index / 8;
        let shift = (bit_index % 8) as u8;
        let v = value & mask;
        out[byte_index] |= v << shift;
        let spill = shift + bits_per;
        if spill > 8 {
            let next_index = byte_index + 1;
            if next_index < out.len() {
                out[next_index] |= v >> (8 - shift);
            }
        }
    }
    out
}

fn unpack_packed_bits(buf: &[u8], bits_per: u8, len: usize) -> Result<Vec<u8>> {
    if bits_per == 0 || bits_per > 8 {
        return Err(anyhow!("invalid packed bit width {}", bits_per));
    }
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        let bit_index = idx * bits_per as usize;
        let byte_index = bit_index / 8;
        let shift = (bit_index % 8) as u8;
        if byte_index >= buf.len() {
            return Err(anyhow!("packed tensor data out of bounds"));
        }
        let mut value = (buf[byte_index] >> shift) as u16;
        let remaining = 8u8.saturating_sub(shift);
        if remaining < bits_per {
            if byte_index + 1 >= buf.len() {
                return Err(anyhow!("packed tensor data out of bounds"));
            }
            value |= (buf[byte_index + 1] as u16) << remaining;
        }
        let mask = (1u16 << bits_per) - 1;
        out.push((value & mask) as u8);
    }
    Ok(out)
}
