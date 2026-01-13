use anyhow::{anyhow, Result};

#[cfg(feature = "vulkan")]
use crate::backend::DeviceTensor;
use crate::backend::TensorStorage;
use crate::graph::{OpAttrs, OpKind};
use crate::ops::{broadcast_enabled, lookup_kernel, KernelFn};
use crate::ops::cpu::{abs, add, mul, relu};
#[cfg(feature = "avx")]
use crate::ops::cpu_avx::{abs as abs_avx, add as add_avx, mul as mul_avx, relu as relu_avx};
#[cfg(feature = "avx2")]
use crate::ops::cpu_avx2::{abs as abs_avx2, add as add_avx2, mul as mul_avx2, relu as relu_avx2};
use crate::simulator::{Device, DeviceBackend};
use crate::tensor::{broadcast_shapes, broadcast_value_to_shape, DType, TensorValue};

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
                    .map(|value| broadcast_value_to_shape(value, &out_shape))
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
        let _ = output_dtype;
        let mut host = to_host_tensors(tensors)?;
        if host.is_empty() {
            return Err(anyhow!("inplace op {} expects at least 1 input", op.as_str()));
        }
        let mut output = host.remove(0);
        match op {
            OpKind::Add => match self.device {
                Device::Cpu => inplace_binary(&mut output, &host, thread_id, add_inplace_dispatch)?,
                #[cfg(feature = "avx")]
                Device::CpuAvx => inplace_binary(&mut output, &host, thread_id, add_inplace_dispatch_avx)?,
                #[cfg(feature = "avx2")]
                Device::CpuAvx2 => inplace_binary(&mut output, &host, thread_id, add_inplace_dispatch_avx2)?,
                _ => return Err(anyhow!("inplace add not supported on {:?}", self.device)),
            },
            OpKind::Mul => match self.device {
                Device::Cpu => inplace_binary(&mut output, &host, thread_id, mul_inplace_dispatch)?,
                #[cfg(feature = "avx")]
                Device::CpuAvx => inplace_binary(&mut output, &host, thread_id, mul_inplace_dispatch_avx)?,
                #[cfg(feature = "avx2")]
                Device::CpuAvx2 => inplace_binary(&mut output, &host, thread_id, mul_inplace_dispatch_avx2)?,
                _ => return Err(anyhow!("inplace mul not supported on {:?}", self.device)),
            },
            OpKind::Abs => match self.device {
                Device::Cpu => inplace_unary(&mut output, thread_id, abs_inplace_dispatch)?,
                #[cfg(feature = "avx")]
                Device::CpuAvx => inplace_unary(&mut output, thread_id, abs_inplace_dispatch_avx)?,
                #[cfg(feature = "avx2")]
                Device::CpuAvx2 => inplace_unary(&mut output, thread_id, abs_inplace_dispatch_avx2)?,
                _ => return Err(anyhow!("inplace abs not supported on {:?}", self.device)),
            },
            OpKind::Relu => match self.device {
                Device::Cpu => inplace_unary_attrs(&mut output, attrs, thread_id, relu_inplace_dispatch)?,
                #[cfg(feature = "avx")]
                Device::CpuAvx => {
                    inplace_unary_attrs(&mut output, attrs, thread_id, relu_inplace_dispatch_avx)?
                }
                #[cfg(feature = "avx2")]
                Device::CpuAvx2 => {
                    inplace_unary_attrs(&mut output, attrs, thread_id, relu_inplace_dispatch_avx2)?
                }
                _ => return Err(anyhow!("inplace relu not supported on {:?}", self.device)),
            },
        }
        Ok(TensorStorage::Host(output))
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

fn inplace_binary(
    output: &mut TensorValue,
    inputs: &[TensorValue],
    thread_id: usize,
    func: fn(&mut TensorValue, &TensorValue, usize) -> Result<()>,
) -> Result<()> {
    let other = inputs
        .get(0)
        .ok_or_else(|| anyhow!("inplace binary op expects at least 1 input"))?;
    func(output, other, thread_id)
}

fn inplace_unary(
    output: &mut TensorValue,
    thread_id: usize,
    func: fn(&mut TensorValue, usize) -> Result<()>,
) -> Result<()> {
    func(output, thread_id)
}

fn inplace_unary_attrs(
    output: &mut TensorValue,
    attrs: &OpAttrs,
    thread_id: usize,
    func: fn(&mut TensorValue, &OpAttrs, usize) -> Result<()>,
) -> Result<()> {
    func(output, attrs, thread_id)
}

fn add_inplace_dispatch(
    output: &mut TensorValue,
    other: &TensorValue,
    thread_id: usize,
) -> Result<()> {
    match (output, other) {
        (TensorValue::I8(out), TensorValue::I8(b)) => add::add_inplace_i8(&mut out.data, &b.data, thread_id),
        (TensorValue::I16(out), TensorValue::I16(b)) => add::add_inplace_i16(&mut out.data, &b.data, thread_id),
        (TensorValue::F32(out), TensorValue::F32(b)) => add::add_inplace_f32(&mut out.data, &b.data, thread_id),
        (TensorValue::F64(out), TensorValue::F64(b)) => add::add_inplace_f64(&mut out.data, &b.data, thread_id),
        (TensorValue::U8(out), TensorValue::U8(b)) => add::add_inplace_u8(&mut out.data, &b.data, thread_id),
        (TensorValue::U16(out), TensorValue::U16(b)) => add::add_inplace_u16(&mut out.data, &b.data, thread_id),
        (TensorValue::I32(out), TensorValue::I32(b)) => add::add_inplace_i32(&mut out.data, &b.data, thread_id),
        (TensorValue::I64(out), TensorValue::I64(b)) => add::add_inplace_i64(&mut out.data, &b.data, thread_id),
        (TensorValue::U32(out), TensorValue::U32(b)) => add::add_inplace_u32(&mut out.data, &b.data, thread_id),
        (TensorValue::U64(out), TensorValue::U64(b)) => add::add_inplace_u64(&mut out.data, &b.data, thread_id),
        (TensorValue::Bool(out), TensorValue::Bool(b)) => add::add_inplace_bool(&mut out.data, &b.data, thread_id),
        _ => Err(anyhow!("add inplace dtype mismatch")),
    }
}

fn mul_inplace_dispatch(
    output: &mut TensorValue,
    other: &TensorValue,
    thread_id: usize,
) -> Result<()> {
    match (output, other) {
        (TensorValue::I8(out), TensorValue::I8(b)) => mul::mul_inplace_i8(&mut out.data, &b.data, thread_id),
        (TensorValue::I16(out), TensorValue::I16(b)) => mul::mul_inplace_i16(&mut out.data, &b.data, thread_id),
        (TensorValue::F32(out), TensorValue::F32(b)) => mul::mul_inplace_f32(&mut out.data, &b.data, thread_id),
        (TensorValue::F64(out), TensorValue::F64(b)) => mul::mul_inplace_f64(&mut out.data, &b.data, thread_id),
        (TensorValue::U8(out), TensorValue::U8(b)) => mul::mul_inplace_u8(&mut out.data, &b.data, thread_id),
        (TensorValue::U16(out), TensorValue::U16(b)) => mul::mul_inplace_u16(&mut out.data, &b.data, thread_id),
        (TensorValue::I32(out), TensorValue::I32(b)) => mul::mul_inplace_i32(&mut out.data, &b.data, thread_id),
        (TensorValue::I64(out), TensorValue::I64(b)) => mul::mul_inplace_i64(&mut out.data, &b.data, thread_id),
        (TensorValue::U32(out), TensorValue::U32(b)) => mul::mul_inplace_u32(&mut out.data, &b.data, thread_id),
        (TensorValue::U64(out), TensorValue::U64(b)) => mul::mul_inplace_u64(&mut out.data, &b.data, thread_id),
        (TensorValue::Bool(out), TensorValue::Bool(b)) => mul::mul_inplace_bool(&mut out.data, &b.data, thread_id),
        _ => Err(anyhow!("mul inplace dtype mismatch")),
    }
}

fn abs_inplace_dispatch(output: &mut TensorValue, thread_id: usize) -> Result<()> {
    match output {
        TensorValue::I8(out) => abs::abs_inplace_i8(&mut out.data, thread_id),
        TensorValue::I16(out) => abs::abs_inplace_i16(&mut out.data, thread_id),
        TensorValue::F32(out) => abs::abs_inplace_f32(&mut out.data, thread_id),
        TensorValue::F64(out) => abs::abs_inplace_f64(&mut out.data, thread_id),
        TensorValue::U8(out) => abs::abs_inplace_u8(&mut out.data, thread_id),
        TensorValue::U16(out) => abs::abs_inplace_u16(&mut out.data, thread_id),
        TensorValue::I32(out) => abs::abs_inplace_i32(&mut out.data, thread_id),
        TensorValue::I64(out) => abs::abs_inplace_i64(&mut out.data, thread_id),
        TensorValue::U32(out) => abs::abs_inplace_u32(&mut out.data, thread_id),
        TensorValue::U64(out) => abs::abs_inplace_u64(&mut out.data, thread_id),
        TensorValue::Bool(out) => abs::abs_inplace_bool(&mut out.data, thread_id),
        _ => Err(anyhow!("abs inplace dtype mismatch")),
    }
}

fn relu_inplace_dispatch(
    output: &mut TensorValue,
    attrs: &OpAttrs,
    thread_id: usize,
) -> Result<()> {
    match output {
        TensorValue::I8(out) => relu::relu_inplace_i8(attrs, &mut out.data, thread_id),
        TensorValue::I16(out) => relu::relu_inplace_i16(attrs, &mut out.data, thread_id),
        TensorValue::F32(out) => relu::relu_inplace_f32(attrs, &mut out.data, thread_id),
        TensorValue::F64(out) => relu::relu_inplace_f64(attrs, &mut out.data, thread_id),
        TensorValue::U8(out) => relu::relu_inplace_u8(attrs, &mut out.data, thread_id),
        TensorValue::U16(out) => relu::relu_inplace_u16(attrs, &mut out.data, thread_id),
        TensorValue::I32(out) => relu::relu_inplace_i32(attrs, &mut out.data, thread_id),
        TensorValue::I64(out) => relu::relu_inplace_i64(attrs, &mut out.data, thread_id),
        TensorValue::U32(out) => relu::relu_inplace_u32(attrs, &mut out.data, thread_id),
        TensorValue::U64(out) => relu::relu_inplace_u64(attrs, &mut out.data, thread_id),
        TensorValue::Bool(out) => relu::relu_inplace_bool(attrs, &mut out.data, thread_id),
        _ => Err(anyhow!("relu inplace dtype mismatch")),
    }
}

#[cfg(feature = "avx")]
fn add_inplace_dispatch_avx(
    output: &mut TensorValue,
    other: &TensorValue,
    thread_id: usize,
) -> Result<()> {
    match (output, other) {
        (TensorValue::I8(out), TensorValue::I8(b)) => add_avx::add_inplace_i8(&mut out.data, &b.data, thread_id),
        (TensorValue::I16(out), TensorValue::I16(b)) => add_avx::add_inplace_i16(&mut out.data, &b.data, thread_id),
        (TensorValue::F32(out), TensorValue::F32(b)) => add_avx::add_inplace_f32(&mut out.data, &b.data, thread_id),
        (TensorValue::F64(out), TensorValue::F64(b)) => add_avx::add_inplace_f64(&mut out.data, &b.data, thread_id),
        (TensorValue::U8(out), TensorValue::U8(b)) => add_avx::add_inplace_u8(&mut out.data, &b.data, thread_id),
        (TensorValue::U16(out), TensorValue::U16(b)) => add_avx::add_inplace_u16(&mut out.data, &b.data, thread_id),
        (TensorValue::I32(out), TensorValue::I32(b)) => add_avx::add_inplace_i32(&mut out.data, &b.data, thread_id),
        (TensorValue::I64(out), TensorValue::I64(b)) => add_avx::add_inplace_i64(&mut out.data, &b.data, thread_id),
        (TensorValue::U32(out), TensorValue::U32(b)) => add_avx::add_inplace_u32(&mut out.data, &b.data, thread_id),
        (TensorValue::U64(out), TensorValue::U64(b)) => add_avx::add_inplace_u64(&mut out.data, &b.data, thread_id),
        (TensorValue::Bool(out), TensorValue::Bool(b)) => add_avx::add_inplace_bool(&mut out.data, &b.data, thread_id),
        _ => Err(anyhow!("add inplace dtype mismatch")),
    }
}

#[cfg(feature = "avx2")]
fn add_inplace_dispatch_avx2(
    output: &mut TensorValue,
    other: &TensorValue,
    thread_id: usize,
) -> Result<()> {
    match (output, other) {
        (TensorValue::I8(out), TensorValue::I8(b)) => add_avx2::add_inplace_i8(&mut out.data, &b.data, thread_id),
        (TensorValue::I16(out), TensorValue::I16(b)) => add_avx2::add_inplace_i16(&mut out.data, &b.data, thread_id),
        (TensorValue::F32(out), TensorValue::F32(b)) => add_avx2::add_inplace_f32(&mut out.data, &b.data, thread_id),
        (TensorValue::F64(out), TensorValue::F64(b)) => add_avx2::add_inplace_f64(&mut out.data, &b.data, thread_id),
        (TensorValue::U8(out), TensorValue::U8(b)) => add_avx2::add_inplace_u8(&mut out.data, &b.data, thread_id),
        (TensorValue::U16(out), TensorValue::U16(b)) => add_avx2::add_inplace_u16(&mut out.data, &b.data, thread_id),
        (TensorValue::I32(out), TensorValue::I32(b)) => add_avx2::add_inplace_i32(&mut out.data, &b.data, thread_id),
        (TensorValue::I64(out), TensorValue::I64(b)) => add_avx2::add_inplace_i64(&mut out.data, &b.data, thread_id),
        (TensorValue::U32(out), TensorValue::U32(b)) => add_avx2::add_inplace_u32(&mut out.data, &b.data, thread_id),
        (TensorValue::U64(out), TensorValue::U64(b)) => add_avx2::add_inplace_u64(&mut out.data, &b.data, thread_id),
        (TensorValue::Bool(out), TensorValue::Bool(b)) => add_avx2::add_inplace_bool(&mut out.data, &b.data, thread_id),
        _ => Err(anyhow!("add inplace dtype mismatch")),
    }
}

#[cfg(feature = "avx")]
fn mul_inplace_dispatch_avx(
    output: &mut TensorValue,
    other: &TensorValue,
    thread_id: usize,
) -> Result<()> {
    match (output, other) {
        (TensorValue::I8(out), TensorValue::I8(b)) => mul_avx::mul_inplace_i8(&mut out.data, &b.data, thread_id),
        (TensorValue::I16(out), TensorValue::I16(b)) => mul_avx::mul_inplace_i16(&mut out.data, &b.data, thread_id),
        (TensorValue::F32(out), TensorValue::F32(b)) => mul_avx::mul_inplace_f32(&mut out.data, &b.data, thread_id),
        (TensorValue::F64(out), TensorValue::F64(b)) => mul_avx::mul_inplace_f64(&mut out.data, &b.data, thread_id),
        (TensorValue::U8(out), TensorValue::U8(b)) => mul_avx::mul_inplace_u8(&mut out.data, &b.data, thread_id),
        (TensorValue::U16(out), TensorValue::U16(b)) => mul_avx::mul_inplace_u16(&mut out.data, &b.data, thread_id),
        (TensorValue::I32(out), TensorValue::I32(b)) => mul_avx::mul_inplace_i32(&mut out.data, &b.data, thread_id),
        (TensorValue::I64(out), TensorValue::I64(b)) => mul_avx::mul_inplace_i64(&mut out.data, &b.data, thread_id),
        (TensorValue::U32(out), TensorValue::U32(b)) => mul_avx::mul_inplace_u32(&mut out.data, &b.data, thread_id),
        (TensorValue::U64(out), TensorValue::U64(b)) => mul_avx::mul_inplace_u64(&mut out.data, &b.data, thread_id),
        (TensorValue::Bool(out), TensorValue::Bool(b)) => mul_avx::mul_inplace_bool(&mut out.data, &b.data, thread_id),
        _ => Err(anyhow!("mul inplace dtype mismatch")),
    }
}

#[cfg(feature = "avx2")]
fn mul_inplace_dispatch_avx2(
    output: &mut TensorValue,
    other: &TensorValue,
    thread_id: usize,
) -> Result<()> {
    match (output, other) {
        (TensorValue::I8(out), TensorValue::I8(b)) => mul_avx2::mul_inplace_i8(&mut out.data, &b.data, thread_id),
        (TensorValue::I16(out), TensorValue::I16(b)) => mul_avx2::mul_inplace_i16(&mut out.data, &b.data, thread_id),
        (TensorValue::F32(out), TensorValue::F32(b)) => mul_avx2::mul_inplace_f32(&mut out.data, &b.data, thread_id),
        (TensorValue::F64(out), TensorValue::F64(b)) => mul_avx2::mul_inplace_f64(&mut out.data, &b.data, thread_id),
        (TensorValue::U8(out), TensorValue::U8(b)) => mul_avx2::mul_inplace_u8(&mut out.data, &b.data, thread_id),
        (TensorValue::U16(out), TensorValue::U16(b)) => mul_avx2::mul_inplace_u16(&mut out.data, &b.data, thread_id),
        (TensorValue::I32(out), TensorValue::I32(b)) => mul_avx2::mul_inplace_i32(&mut out.data, &b.data, thread_id),
        (TensorValue::I64(out), TensorValue::I64(b)) => mul_avx2::mul_inplace_i64(&mut out.data, &b.data, thread_id),
        (TensorValue::U32(out), TensorValue::U32(b)) => mul_avx2::mul_inplace_u32(&mut out.data, &b.data, thread_id),
        (TensorValue::U64(out), TensorValue::U64(b)) => mul_avx2::mul_inplace_u64(&mut out.data, &b.data, thread_id),
        (TensorValue::Bool(out), TensorValue::Bool(b)) => mul_avx2::mul_inplace_bool(&mut out.data, &b.data, thread_id),
        _ => Err(anyhow!("mul inplace dtype mismatch")),
    }
}

#[cfg(feature = "avx")]
fn abs_inplace_dispatch_avx(output: &mut TensorValue, thread_id: usize) -> Result<()> {
    match output {
        TensorValue::I8(out) => abs_avx::abs_inplace_i8(&mut out.data, thread_id),
        TensorValue::I16(out) => abs_avx::abs_inplace_i16(&mut out.data, thread_id),
        TensorValue::F32(out) => abs_avx::abs_inplace_f32(&mut out.data, thread_id),
        TensorValue::F64(out) => abs_avx::abs_inplace_f64(&mut out.data, thread_id),
        TensorValue::U8(out) => abs_avx::abs_inplace_u8(&mut out.data, thread_id),
        TensorValue::U16(out) => abs_avx::abs_inplace_u16(&mut out.data, thread_id),
        TensorValue::I32(out) => abs_avx::abs_inplace_i32(&mut out.data, thread_id),
        TensorValue::I64(out) => abs_avx::abs_inplace_i64(&mut out.data, thread_id),
        TensorValue::U32(out) => abs_avx::abs_inplace_u32(&mut out.data, thread_id),
        TensorValue::U64(out) => abs_avx::abs_inplace_u64(&mut out.data, thread_id),
        TensorValue::Bool(out) => abs_avx::abs_inplace_bool(&mut out.data, thread_id),
        _ => Err(anyhow!("abs inplace dtype mismatch")),
    }
}

#[cfg(feature = "avx2")]
fn abs_inplace_dispatch_avx2(output: &mut TensorValue, thread_id: usize) -> Result<()> {
    match output {
        TensorValue::I8(out) => abs_avx2::abs_inplace_i8(&mut out.data, thread_id),
        TensorValue::I16(out) => abs_avx2::abs_inplace_i16(&mut out.data, thread_id),
        TensorValue::F32(out) => abs_avx2::abs_inplace_f32(&mut out.data, thread_id),
        TensorValue::F64(out) => abs_avx2::abs_inplace_f64(&mut out.data, thread_id),
        TensorValue::U8(out) => abs_avx2::abs_inplace_u8(&mut out.data, thread_id),
        TensorValue::U16(out) => abs_avx2::abs_inplace_u16(&mut out.data, thread_id),
        TensorValue::I32(out) => abs_avx2::abs_inplace_i32(&mut out.data, thread_id),
        TensorValue::I64(out) => abs_avx2::abs_inplace_i64(&mut out.data, thread_id),
        TensorValue::U32(out) => abs_avx2::abs_inplace_u32(&mut out.data, thread_id),
        TensorValue::U64(out) => abs_avx2::abs_inplace_u64(&mut out.data, thread_id),
        TensorValue::Bool(out) => abs_avx2::abs_inplace_bool(&mut out.data, thread_id),
        _ => Err(anyhow!("abs inplace dtype mismatch")),
    }
}

#[cfg(feature = "avx")]
fn relu_inplace_dispatch_avx(
    output: &mut TensorValue,
    attrs: &OpAttrs,
    thread_id: usize,
) -> Result<()> {
    match output {
        TensorValue::F32(out) => relu_avx::relu_inplace_f32(attrs, &mut out.data, thread_id),
        _ => Err(anyhow!("relu inplace dtype mismatch")),
    }
}

#[cfg(feature = "avx2")]
fn relu_inplace_dispatch_avx2(
    output: &mut TensorValue,
    attrs: &OpAttrs,
    thread_id: usize,
) -> Result<()> {
    match output {
        TensorValue::F32(out) => relu_avx2::relu_inplace_f32(attrs, &mut out.data, thread_id),
        _ => Err(anyhow!("relu inplace dtype mismatch")),
    }
}
