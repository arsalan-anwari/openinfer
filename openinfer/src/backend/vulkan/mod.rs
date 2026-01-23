use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::backend::{DeviceTensor, OpShaderInfo, ShaderRegistry, TensorStorage, VulkanBuffer};
use crate::backend::cpu::CpuBackend;
use crate::graph::{OpAttrs, OpKind};
use crate::ops::{
    broadcast_enabled,
    broadcast_is_elementwise,
    broadcast_requires_materialize,
    lookup_kernel,
    KernelFn,
};
use crate::ops::registry::lookup_kernel_inplace;
use crate::simulator::{Device, DeviceBackend};
use crate::tensor::{broadcast_shapes, BF16, Bitset, DType, F16, F8E5M2, I1, I2, I4, T1, T2, U1, U2, U4, TensorValue};

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

    #[allow(dead_code)]
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
    force_simulated_float: bool,
}

impl VulkanBackend {

    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::new_with_settings(false)
    }

    pub fn new_with_settings(force_simulated_float: bool) -> Self {
        let shaders = VulkanShaderRegistry::load_default().unwrap_or_else(|err| {
            eprintln!("vulkan shaders: {}", err);
            VulkanShaderRegistry::default()
        });
        Self {
            shaders,
            runtime: Mutex::new(None),
            dispatch_lock: Mutex::new(()),
            force_simulated_float,
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
        let runtime = Arc::new(VulkanRuntime::new_with_settings(self.force_simulated_float)?);
        *guard = Some(Arc::clone(&runtime));
        Ok(runtime)
    }

    fn effective_dtype(&self, dtype: DType) -> Result<DType> {
        let runtime = self.runtime()?;
        match dtype {
            DType::F16 | DType::BF16 | DType::F8E5M2 => Ok(dtype),
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
            DType::Bitset => Ok(dtype),
            DType::T1 | DType::T2 => Err(anyhow!("vulkan backend does not support t1/t2 tensors")),
            _ => Ok(dtype)
        }
    }

    fn exec_op_cpu_fallback(
        &self,
        op: OpKind,
        attrs: &OpAttrs,
        output_dtype: DType,
        tensors: &[TensorStorage],
        output: Option<TensorStorage>,
        thread_id: usize,
    ) -> Result<TensorStorage> {
        let cpu_backend = CpuBackend::new(Device::Cpu);
        let mut host_inputs = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            match tensor {
                TensorStorage::Host(value) => host_inputs.push(TensorStorage::Host(value.clone())),
                TensorStorage::Device(DeviceTensor::Vulkan(_)) => {
                    let host = self.download(tensor.clone())?;
                    host_inputs.push(TensorStorage::Host(host));
                }
            }
        }
        let output_host = match output {
            Some(TensorStorage::Host(value)) => Some(TensorStorage::Host(value)),
            Some(TensorStorage::Device(DeviceTensor::Vulkan(buf))) => {
                let host = self.download(TensorStorage::Device(DeviceTensor::Vulkan(buf)))?;
                Some(TensorStorage::Host(host))
            }
            None => None,
        };
        let output = cpu_backend.exec_op(
            op,
            attrs,
            output_dtype,
            &host_inputs,
            output_host,
            thread_id,
        )?;
        match output {
            TensorStorage::Host(value) => match self.effective_dtype(output_dtype) {
                Ok(_) => {
                    eprintln!(
                        "vulkan fallback: running {} on CPU and uploading output",
                        op.as_str()
                    );
                    self.upload(value)
                }
                Err(err) => {
                    eprintln!(
                        "vulkan fallback: running {} on CPU ({}), keeping host output",
                        op.as_str(),
                        err
                    );
                    Ok(TensorStorage::Host(value))
                }
            },
            TensorStorage::Device(_) => Ok(output),
        }
    }

    fn exec_op_inplace_cpu_fallback(
        &self,
        op: OpKind,
        attrs: &OpAttrs,
        output_dtype: DType,
        tensors: &[TensorStorage],
        thread_id: usize,
    ) -> Result<TensorStorage> {
        let cpu_backend = CpuBackend::new(Device::Cpu);
        let mut host_inputs = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            match tensor {
                TensorStorage::Host(value) => host_inputs.push(TensorStorage::Host(value.clone())),
                TensorStorage::Device(DeviceTensor::Vulkan(_)) => {
                    let host = self.download(tensor.clone())?;
                    host_inputs.push(TensorStorage::Host(host));
                }
            }
        }
        let output = cpu_backend.exec_op_inplace(op, attrs, output_dtype, &host_inputs, thread_id)?;
        match output {
            TensorStorage::Host(value) => match self.effective_dtype(output_dtype) {
                Ok(_) => {
                    eprintln!(
                        "vulkan fallback: running {} inplace on CPU and uploading output",
                        op.as_str()
                    );
                    self.upload(value)
                }
                Err(err) => {
                    eprintln!(
                        "vulkan fallback: running {} inplace on CPU ({}), keeping host output",
                        op.as_str(),
                        err
                    );
                    Ok(TensorStorage::Host(value))
                }
            },
            TensorStorage::Device(_) => Ok(output),
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
        let effective_dtype = match self.effective_dtype(dtype) {
            Ok(dtype) => dtype,
            Err(err) => {
                eprintln!("vulkan fallback: alloc {:?} on CPU ({})", dtype, err);
                return Ok(TensorStorage::Host(TensorValue::zeros(dtype, shape)));
            }
        };
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
        let effective_dtype = match self.effective_dtype(dtype) {
            Ok(dtype) => dtype,
            Err(err) => {
                eprintln!("vulkan fallback: upload {:?} on CPU ({})", dtype, err);
                return Ok(TensorStorage::Host(value));
            }
        };
        let len = value.len();
        let shape = value.shape().to_vec();
        let strides = value.strides().to_vec();
        let size = storage_size_bytes_for_len(effective_dtype, len);
        let inner = runtime.create_buffer(size)?;
        let bytes = encode_tensor_with_effective(&value, effective_dtype)?;
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
            TensorStorage::Host(host) => Ok(host),
            TensorStorage::Device(DeviceTensor::Vulkan(buf)) => {
                let runtime = self.runtime()?;
                let size = storage_size_bytes_for_len(buf.effective_dtype, buf.len);
                let mut bytes = vec![0u8; size];
                runtime.read_buffer(&buf.inner, &mut bytes)?;
                decode_tensor_with_effective(
                    buf.dtype,
                    buf.effective_dtype,
                    buf.shape.as_slice(),
                    &bytes,
                )
            }
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
        if tensors.iter().any(|t| matches!(t, TensorStorage::Host(_))) {
            return self.exec_op_cpu_fallback(op, attrs, output_dtype, tensors, output, thread_id);
        }
        let output_effective = match self.effective_dtype(output_dtype) {
            Ok(dtype) => dtype,
            Err(err) => {
                eprintln!(
                    "vulkan fallback: op {} output {:?} unsupported ({})",
                    op.as_str(),
                    output_dtype,
                    err
                );
                return self.exec_op_cpu_fallback(op, attrs, output_dtype, tensors, output, thread_id);
            }
        };
        let shader = self.shaders.shader_for_op(op);
        if shader.is_none() {
            eprintln!("vulkan shaders: missing entry for op {}", op.as_str());
        }
        let mut buffers = to_vulkan_buffers(tensors, shader.clone())?;
        let input_dtypes: Vec<DType> = buffers.iter().map(|b| b.effective_dtype).collect();
        let output_buffer = match output.as_ref() {
            Some(TensorStorage::Device(DeviceTensor::Vulkan(buf))) => Some(buf),
            _ => None,
        };
        let use_accumulate = matches!(attrs, OpAttrs::Accumulate { .. })
            && output_buffer.is_some()
            && matches!(op, OpKind::Add | OpKind::Mul | OpKind::Abs | OpKind::Matmul);
        let kernel = if use_accumulate {
            None
        } else {
            match lookup_kernel(self.device(), op, output_effective, &input_dtypes, attrs) {
                Some(kernel) => Some(kernel),
                None => {
                    eprintln!(
                        "vulkan fallback: no kernel for {} {:?}->{:?}",
                        op.as_str(),
                        input_dtypes,
                        output_effective
                    );
                    return self.exec_op_cpu_fallback(
                        op,
                        attrs,
                        output_dtype,
                        tensors,
                        output,
                        thread_id,
                    );
                }
            }
        };
        if let Some(KernelFn::Host(_)) = kernel {
            eprintln!("vulkan fallback: host kernel for {}", op.as_str());
            return self.exec_op_cpu_fallback(op, attrs, output_dtype, tensors, output, thread_id);
        }
        let mut broadcast_shape = None;
        if buffers.len() > 1
            && broadcast_enabled(op, self.device())
            && broadcast_is_elementwise(op)
            && broadcast_requires_materialize(op)
        {
            let mut out_shape = buffers[0].shape.clone();
            for buffer in buffers.iter().skip(1) {
                out_shape = broadcast_shapes(&out_shape, buffer.shape.as_slice())?;
            }
            if buffers.iter().any(|buffer| buffer.shape != out_shape) {
                broadcast_shape = Some(out_shape);
            }
        }
        let _guard = self
            .dispatch_lock
            .lock()
            .map_err(|_| anyhow!("vulkan dispatch lock poisoned"))?;
        if let Some(out_shape) = broadcast_shape {
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
        let buffer_refs: Vec<&VulkanBuffer> = buffers.iter().collect();
        if use_accumulate {
            let out_buf = output_buffer.expect("accumulate output buffer must exist");
            let output = match op {
                OpKind::Add => crate::ops::vulkan::add::add_accumulate_generic(
                    attrs,
                    &buffers[0],
                    &buffers[1],
                    output_dtype,
                    Some(out_buf),
                    thread_id,
                )?,
                OpKind::Mul => crate::ops::vulkan::mul::mul_accumulate_generic(
                    attrs,
                    &buffers[0],
                    &buffers[1],
                    output_dtype,
                    Some(out_buf),
                    thread_id,
                )?,
                OpKind::Abs => crate::ops::vulkan::abs::abs_accumulate_generic(
                    attrs,
                    &buffers[0],
                    output_dtype,
                    Some(out_buf),
                    thread_id,
                )?,
                OpKind::Matmul => crate::ops::vulkan::matmul::matmul_accumulate_generic(
                    attrs,
                    &buffers[0],
                    &buffers[1],
                    output_dtype,
                    Some(out_buf),
                    thread_id,
                )?,
                _ => unreachable!("unsupported accumulate op"),
            };
            return Ok(TensorStorage::Device(DeviceTensor::Vulkan(output)));
        }
        match kernel.expect("kernel must be resolved before dispatch") {
            KernelFn::Vulkan(func) => Ok(TensorStorage::Device(DeviceTensor::Vulkan(
                (func)(attrs, &buffer_refs, thread_id)?,
            ))),
            KernelFn::Host(_) => unreachable!("host kernel handled before dispatch"),
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
        if tensors.iter().any(|t| matches!(t, TensorStorage::Host(_))) {
            return self.exec_op_inplace_cpu_fallback(op, attrs, output_dtype, tensors, thread_id);
        }
        let output_effective = match self.effective_dtype(output_dtype) {
            Ok(dtype) => dtype,
            Err(err) => {
                eprintln!(
                    "vulkan fallback: inplace op {} output {:?} unsupported ({})",
                    op.as_str(),
                    output_dtype,
                    err
                );
                return self.exec_op_inplace_cpu_fallback(op, attrs, output_dtype, tensors, thread_id);
            }
        };
        let shader = self.shaders.shader_for_op(op);
        if shader.is_none() {
            eprintln!(
                "vulkan shaders: missing entry for op {}",
                op.as_str()
            );
            return self.exec_op_inplace_cpu_fallback(op, attrs, output_dtype, tensors, thread_id);
        }
        let mut buffers = to_vulkan_buffers(tensors, shader)?;
        if buffers.len() > 1 && broadcast_is_elementwise(op) {
            if broadcast_enabled(op, self.device()) {
                let out_shape = buffers[0].shape.clone();
                for buffer in buffers.iter().skip(1) {
                    let merged = broadcast_shapes(&out_shape, buffer.shape.as_slice())?;
                    if merged != out_shape {
                        return Err(anyhow!(
                            "op {} inplace requires broadcastable input shapes on {:?}",
                            op.as_str(),
                            self.device()
                        ));
                    }
                }
                if buffers.iter().skip(1).any(|buffer| buffer.shape != out_shape) {
                    if !matches!(op, OpKind::Add | OpKind::Mul) {
                        let _guard = self
                            .dispatch_lock
                            .lock()
                            .map_err(|_| anyhow!("vulkan dispatch lock poisoned"))?;
                        buffers = buffers
                            .into_iter()
                            .enumerate()
                            .map(|(idx, buffer)| {
                                if idx == 0 || buffer.shape == out_shape {
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
            } else {
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
        }
        let input_dtypes: Vec<DType> = buffers.iter().map(|b| b.effective_dtype).collect();
        let kernel = match lookup_kernel_inplace(self.device(), op, output_effective, &input_dtypes, attrs) {
            Some(kernel) => kernel,
            None => {
                eprintln!(
                    "vulkan fallback: no inplace kernel for {} {:?}->{:?}",
                    op.as_str(),
                    input_dtypes,
                    output_effective
                );
                return self.exec_op_inplace_cpu_fallback(op, attrs, output_dtype, tensors, thread_id);
            }
        };
        match kernel {
            crate::ops::registry::InplaceKernelFn::Host(_) => {
                eprintln!("vulkan fallback: host inplace kernel for {}", op.as_str());
                self.exec_op_inplace_cpu_fallback(op, attrs, output_dtype, tensors, thread_id)
            }
            crate::ops::registry::InplaceKernelFn::Vulkan(func) => {
                let _guard = self
                    .dispatch_lock
                    .lock()
                    .map_err(|_| anyhow!("vulkan dispatch lock poisoned"))?;
                let buffer_refs: Vec<&VulkanBuffer> = buffers.iter().collect();
                Ok(TensorStorage::Device(DeviceTensor::Vulkan(
                (func)(attrs, &buffer_refs, thread_id)?,
                )))
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

fn encode_tensor(value: &TensorValue) -> Vec<u8> {
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
            .flat_map(|v| v.bits.to_le_bytes())
            .collect(),
        TensorValue::BF16(tensor) => tensor
            .data
            .iter()
            .flat_map(|v| v.bits.to_le_bytes())
            .collect(),
        TensorValue::F8E5M2(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
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
        TensorValue::I4(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::I2(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::I1(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::U4(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::U2(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::U1(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::T2(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
        TensorValue::T1(tensor) => tensor.data.iter().map(|v| v.bits).collect(),
    }
}

fn encode_tensor_with_effective(value: &TensorValue, effective: DType) -> Result<Vec<u8>> {
    if value.dtype() == effective {
        return Ok(encode_tensor(value));
    }
    match (value, effective) {
        (TensorValue::F16(tensor), DType::F32) => Ok(tensor
            .data
            .iter()
            .flat_map(|v| v.to_f32().to_le_bytes())
            .collect()),
        (TensorValue::BF16(tensor), DType::F32) => Ok(tensor
            .data
            .iter()
            .flat_map(|v| v.to_f32().to_le_bytes())
            .collect()),
        (TensorValue::F8E5M2(tensor), DType::F32) => Ok(tensor
            .data
            .iter()
            .flat_map(|v| v.to_f32().to_le_bytes())
            .collect()),
        _ => Err(anyhow!(
            "vulkan encode {:?} -> {:?} not supported",
            value.dtype(),
            effective
        )),
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
                .chunks_exact(2)
                .take(len)
                .map(|chunk| F16 {
                    bits: u16::from_le_bytes(chunk.try_into().unwrap()),
                })
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::BF16 => Ok(TensorValue::BF16(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(2)
                .take(len)
                .map(|chunk| BF16 {
                    bits: u16::from_le_bytes(chunk.try_into().unwrap()),
                })
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::F8E5M2 => Ok(TensorValue::F8E5M2(crate::tensor::Tensor::from_vec_with_opts(
            bytes.iter().take(len).map(|bits| F8E5M2 { bits: *bits }).collect(),
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
            Ok(TensorValue::I4(crate::tensor::Tensor::from_vec_with_opts(
                bytes.iter().map(|bits| I4 { bits: *bits }).collect(),
                crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    allow_len_mismatch: true,
                    ..crate::tensor::TensorOptions::default()
                },
            )?))
        }
        DType::I2 => {
            Ok(TensorValue::I2(crate::tensor::Tensor::from_vec_with_opts(
                bytes.iter().map(|bits| I2 { bits: *bits }).collect(),
                crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    allow_len_mismatch: true,
                    ..crate::tensor::TensorOptions::default()
                },
            )?))
        }
        DType::I1 => {
            Ok(TensorValue::I1(crate::tensor::Tensor::from_vec_with_opts(
                bytes.iter().map(|bits| I1 { bits: *bits }).collect(),
                crate::tensor::TensorOptions {
                    shape: Some(shape.to_vec()),
                    allow_len_mismatch: true,
                    ..crate::tensor::TensorOptions::default()
                },
            )?))
        }
        DType::U4 => Ok(TensorValue::U4(crate::tensor::Tensor::from_vec_with_opts(
            bytes.iter().map(|bits| U4 { bits: *bits }).collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                allow_len_mismatch: true,
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::U2 => Ok(TensorValue::U2(crate::tensor::Tensor::from_vec_with_opts(
            bytes.iter().map(|bits| U2 { bits: *bits }).collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                allow_len_mismatch: true,
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::U1 => Ok(TensorValue::U1(crate::tensor::Tensor::from_vec_with_opts(
            bytes.iter().map(|bits| U1 { bits: *bits }).collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                allow_len_mismatch: true,
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::T2 => Ok(TensorValue::T2(crate::tensor::Tensor::from_vec_with_opts(
            bytes.iter().map(|bits| T2 { bits: *bits }).collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                allow_len_mismatch: true,
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        DType::T1 => Ok(TensorValue::T1(crate::tensor::Tensor::from_vec_with_opts(
            bytes.iter().map(|bits| T1 { bits: *bits }).collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                allow_len_mismatch: true,
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
    }
}

fn decode_tensor_with_effective(
    original: DType,
    effective: DType,
    shape: &[usize],
    bytes: &[u8],
) -> Result<TensorValue> {
    if original == effective {
        return decode_tensor(effective, shape, bytes);
    }
    let len = crate::tensor::numel(shape);
    match (original, effective) {
        (DType::F16, DType::F32) => Ok(TensorValue::F16(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| F16::from_f32(f32::from_le_bytes(chunk.try_into().unwrap())))
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        (DType::BF16, DType::F32) => Ok(TensorValue::BF16(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| BF16::from_f32(f32::from_le_bytes(chunk.try_into().unwrap())))
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        (DType::F8E5M2, DType::F32) => Ok(TensorValue::F8E5M2(crate::tensor::Tensor::from_vec_with_opts(
            bytes
                .chunks_exact(4)
                .take(len)
                .map(|chunk| F8E5M2::from_f32(f32::from_le_bytes(chunk.try_into().unwrap())))
                .collect(),
            crate::tensor::TensorOptions {
                shape: Some(shape.to_vec()),
                ..crate::tensor::TensorOptions::default()
            },
        )?)),
        _ => Err(anyhow!(
            "vulkan decode {:?} <- {:?} not supported",
            original,
            effective
        )),
    }
}
