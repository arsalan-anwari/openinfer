use anyhow::{anyhow, Result};

use crate::backend::VulkanBuffer;
use crate::graph::{OpAttrs, OpKind};
use crate::tensor::{DType, TensorElement, TensorValue};
use crate::executor::Device;

use super::{abs_f32, add_f32, mul_f32};

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
use super::{abs_f32_avx, add_f32_avx, mul_f32_avx};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use super::{abs_f32_avx2, add_f32_avx2, mul_f32_avx2};

#[cfg(feature = "vulkan")]
use super::{abs_f32_vulkan, add_f32_vulkan, mul_f32_vulkan};

pub type HostKernel = Box<dyn Fn(&OpAttrs, &[TensorValue]) -> Result<TensorValue> + Send + Sync>;
pub type VulkanKernel =
    Box<dyn Fn(&OpAttrs, &[&VulkanBuffer]) -> Result<VulkanBuffer> + Send + Sync>;

pub trait CpuKernelAdapter {
    fn call(&self, attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue>;
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
pub trait DeviceKernelAdapter {
    fn call(&self, attrs: &OpAttrs, inputs: &[&VulkanBuffer]) -> Result<VulkanBuffer>;
}

pub fn cpu_kernel<K>(func: K) -> HostKernel
where
    K: CpuKernelAdapter + Send + Sync + 'static,
{
    Box::new(move |attrs, inputs| func.call(attrs, inputs))
}

#[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
pub fn device_kernel<K>(func: K) -> VulkanKernel
where
    K: DeviceKernelAdapter + Send + Sync + 'static,
{
    Box::new(move |attrs, inputs| func.call(attrs, inputs))
}

pub enum KernelFn {
    Host(HostKernel),
    #[cfg_attr(not(feature = "vulkan"), allow(dead_code))]
    Vulkan(VulkanKernel),
}

pub fn lookup_kernel(
    device: Device,
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: OpAttrs,
) -> Option<KernelFn> {
    match (device, op, output_dtype, input_dtypes, attrs) {
        (Device::Cpu, OpKind::Add, DType::F32, [DType::F32, DType::F32], OpAttrs::None) => {
            Some(KernelFn::Host(cpu_kernel(
                add_f32 as fn(&[f32], &[f32]) -> Result<Vec<f32>>,
            )))
        }
        (Device::Cpu, OpKind::Mul, DType::F32, [DType::F32, DType::F32], OpAttrs::None) => {
            Some(KernelFn::Host(cpu_kernel(
                mul_f32 as fn(&[f32], &[f32]) -> Result<Vec<f32>>,
            )))
        }
        (Device::Cpu, OpKind::Abs, DType::F32, [DType::F32], OpAttrs::None) => {
            Some(KernelFn::Host(cpu_kernel(
                abs_f32 as fn(&[f32]) -> Result<Vec<f32>>,
            )))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        (Device::CpuAvx, OpKind::Add, DType::F32, [DType::F32, DType::F32], OpAttrs::None) => {
            Some(KernelFn::Host(cpu_kernel(
                add_f32_avx as fn(&[f32], &[f32]) -> Result<Vec<f32>>,
            )))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        (Device::CpuAvx, OpKind::Mul, DType::F32, [DType::F32, DType::F32], OpAttrs::None) => {
            Some(KernelFn::Host(cpu_kernel(
                mul_f32_avx as fn(&[f32], &[f32]) -> Result<Vec<f32>>,
            )))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        (Device::CpuAvx, OpKind::Abs, DType::F32, [DType::F32], OpAttrs::None) => {
            Some(KernelFn::Host(cpu_kernel(
                abs_f32_avx as fn(&[f32]) -> Result<Vec<f32>>,
            )))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        (Device::CpuAvx2, OpKind::Add, DType::F32, [DType::F32, DType::F32], OpAttrs::None) => {
            Some(KernelFn::Host(cpu_kernel(
                add_f32_avx2 as fn(&[f32], &[f32]) -> Result<Vec<f32>>,
            )))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        (Device::CpuAvx2, OpKind::Mul, DType::F32, [DType::F32, DType::F32], OpAttrs::None) => {
            Some(KernelFn::Host(cpu_kernel(
                mul_f32_avx2 as fn(&[f32], &[f32]) -> Result<Vec<f32>>,
            )))
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        (Device::CpuAvx2, OpKind::Abs, DType::F32, [DType::F32], OpAttrs::None) => {
            Some(KernelFn::Host(cpu_kernel(
                abs_f32_avx2 as fn(&[f32]) -> Result<Vec<f32>>,
            )))
        }
        #[cfg(feature = "vulkan")]
        (Device::Vulkan, OpKind::Add, DType::F32, [DType::F32, DType::F32], OpAttrs::None) => {
            Some(KernelFn::Vulkan(device_kernel(
                add_f32_vulkan as fn(&VulkanBuffer, &VulkanBuffer) -> Result<VulkanBuffer>,
            )))
        }
        #[cfg(feature = "vulkan")]
        (Device::Vulkan, OpKind::Mul, DType::F32, [DType::F32, DType::F32], OpAttrs::None) => {
            Some(KernelFn::Vulkan(device_kernel(
                mul_f32_vulkan as fn(&VulkanBuffer, &VulkanBuffer) -> Result<VulkanBuffer>,
            )))
        }
        #[cfg(feature = "vulkan")]
        (Device::Vulkan, OpKind::Abs, DType::F32, [DType::F32], OpAttrs::None) => {
            Some(KernelFn::Vulkan(device_kernel(
                abs_f32_vulkan as fn(&VulkanBuffer) -> Result<VulkanBuffer>,
            )))
        }
        _ => None,
    }
}

impl<A, R> CpuKernelAdapter for for<'a> fn(&'a [A]) -> Result<R>
where
    A: TensorElement + Clone + 'static,
    R: Into<TensorValue>,
{
    fn call(&self, _attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
        let a = inputs
            .get(0)
            .ok_or_else(|| anyhow!("cpu kernel expects at least 1 input"))?;
        let a = A::from_value(a)
            .ok_or_else(|| anyhow!("cpu kernel input 0 has wrong dtype"))?;
        Ok((self)(&a.data)?.into())
    }
}

impl<A, B, R> CpuKernelAdapter for for<'a, 'b> fn(&'a [A], &'b [B]) -> Result<R>
where
    A: TensorElement + Clone + 'static,
    B: TensorElement + Clone + 'static,
    R: Into<TensorValue>,
{
    fn call(&self, _attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
        if inputs.len() < 2 {
            return Err(anyhow!("cpu kernel expects at least 2 inputs"));
        }
        let a = A::from_value(&inputs[0])
            .ok_or_else(|| anyhow!("cpu kernel input 0 has wrong dtype"))?;
        let b = B::from_value(&inputs[1])
            .ok_or_else(|| anyhow!("cpu kernel input 1 has wrong dtype"))?;
        Ok((self)(&a.data, &b.data)?.into())
    }
}

impl<A, B, C, R> CpuKernelAdapter
    for for<'a, 'b, 'c> fn(&'a [A], &'b [B], &'c [C]) -> Result<R>
where
    A: TensorElement + Clone + 'static,
    B: TensorElement + Clone + 'static,
    C: TensorElement + Clone + 'static,
    R: Into<TensorValue>,
{
    fn call(&self, _attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
        if inputs.len() < 3 {
            return Err(anyhow!("cpu kernel expects at least 3 inputs"));
        }
        let a = A::from_value(&inputs[0])
            .ok_or_else(|| anyhow!("cpu kernel input 0 has wrong dtype"))?;
        let b = B::from_value(&inputs[1])
            .ok_or_else(|| anyhow!("cpu kernel input 1 has wrong dtype"))?;
        let c = C::from_value(&inputs[2])
            .ok_or_else(|| anyhow!("cpu kernel input 2 has wrong dtype"))?;
        Ok((self)(&a.data, &b.data, &c.data)?.into())
    }
}

impl<A, B, C, D, R> CpuKernelAdapter
    for for<'a, 'b, 'c, 'd> fn(&'a [A], &'b [B], &'c [C], &'d [D]) -> Result<R>
where
    A: TensorElement + Clone + 'static,
    B: TensorElement + Clone + 'static,
    C: TensorElement + Clone + 'static,
    D: TensorElement + Clone + 'static,
    R: Into<TensorValue>,
{
    fn call(&self, _attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
        if inputs.len() < 4 {
            return Err(anyhow!("cpu kernel expects at least 4 inputs"));
        }
        let a = A::from_value(&inputs[0])
            .ok_or_else(|| anyhow!("cpu kernel input 0 has wrong dtype"))?;
        let b = B::from_value(&inputs[1])
            .ok_or_else(|| anyhow!("cpu kernel input 1 has wrong dtype"))?;
        let c = C::from_value(&inputs[2])
            .ok_or_else(|| anyhow!("cpu kernel input 2 has wrong dtype"))?;
        let d = D::from_value(&inputs[3])
            .ok_or_else(|| anyhow!("cpu kernel input 3 has wrong dtype"))?;
        Ok((self)(&a.data, &b.data, &c.data, &d.data)?.into())
    }
}

impl<A, R> CpuKernelAdapter for for<'a, 'b> fn(&'a OpAttrs, &'b [A]) -> Result<R>
where
    A: TensorElement + Clone + 'static,
    R: Into<TensorValue>,
{
    fn call(&self, attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
        let a = inputs
            .get(0)
            .ok_or_else(|| anyhow!("cpu kernel expects at least 1 input"))?;
        let a = A::from_value(a)
            .ok_or_else(|| anyhow!("cpu kernel input 0 has wrong dtype"))?;
        Ok((self)(attrs, &a.data)?.into())
    }
}

impl<A, B, R> CpuKernelAdapter
    for for<'a, 'b, 'c> fn(&'a OpAttrs, &'b [A], &'c [B]) -> Result<R>
where
    A: TensorElement + Clone + 'static,
    B: TensorElement + Clone + 'static,
    R: Into<TensorValue>,
{
    fn call(&self, attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
        if inputs.len() < 2 {
            return Err(anyhow!("cpu kernel expects at least 2 inputs"));
        }
        let a = A::from_value(&inputs[0])
            .ok_or_else(|| anyhow!("cpu kernel input 0 has wrong dtype"))?;
        let b = B::from_value(&inputs[1])
            .ok_or_else(|| anyhow!("cpu kernel input 1 has wrong dtype"))?;
        Ok((self)(attrs, &a.data, &b.data)?.into())
    }
}

impl<A, B, C, R> CpuKernelAdapter
    for for<'a, 'b, 'c, 'd> fn(&'a OpAttrs, &'b [A], &'c [B], &'d [C]) -> Result<R>
where
    A: TensorElement + Clone + 'static,
    B: TensorElement + Clone + 'static,
    C: TensorElement + Clone + 'static,
    R: Into<TensorValue>,
{
    fn call(&self, attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
        if inputs.len() < 3 {
            return Err(anyhow!("cpu kernel expects at least 3 inputs"));
        }
        let a = A::from_value(&inputs[0])
            .ok_or_else(|| anyhow!("cpu kernel input 0 has wrong dtype"))?;
        let b = B::from_value(&inputs[1])
            .ok_or_else(|| anyhow!("cpu kernel input 1 has wrong dtype"))?;
        let c = C::from_value(&inputs[2])
            .ok_or_else(|| anyhow!("cpu kernel input 2 has wrong dtype"))?;
        Ok((self)(attrs, &a.data, &b.data, &c.data)?.into())
    }
}

impl<A, B, C, D, R> CpuKernelAdapter
    for for<'a, 'b, 'c, 'd, 'e> fn(&'a OpAttrs, &'b [A], &'c [B], &'d [C], &'e [D]) -> Result<R>
where
    A: TensorElement + Clone + 'static,
    B: TensorElement + Clone + 'static,
    C: TensorElement + Clone + 'static,
    D: TensorElement + Clone + 'static,
    R: Into<TensorValue>,
{
    fn call(&self, attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
        if inputs.len() < 4 {
            return Err(anyhow!("cpu kernel expects at least 4 inputs"));
        }
        let a = A::from_value(&inputs[0])
            .ok_or_else(|| anyhow!("cpu kernel input 0 has wrong dtype"))?;
        let b = B::from_value(&inputs[1])
            .ok_or_else(|| anyhow!("cpu kernel input 1 has wrong dtype"))?;
        let c = C::from_value(&inputs[2])
            .ok_or_else(|| anyhow!("cpu kernel input 2 has wrong dtype"))?;
        let d = D::from_value(&inputs[3])
            .ok_or_else(|| anyhow!("cpu kernel input 3 has wrong dtype"))?;
        Ok((self)(attrs, &a.data, &b.data, &c.data, &d.data)?.into())
    }
}

impl<R> DeviceKernelAdapter for for<'a> fn(&'a VulkanBuffer) -> Result<R>
where
    R: Into<VulkanBuffer>,
{
    fn call(&self, _attrs: &OpAttrs, inputs: &[&VulkanBuffer]) -> Result<VulkanBuffer> {
        let a = inputs
            .get(0)
            .ok_or_else(|| anyhow!("device kernel expects at least 1 input"))?;
        Ok((self)(a)?.into())
    }
}

impl<R> DeviceKernelAdapter
    for for<'a, 'b> fn(&'a VulkanBuffer, &'b VulkanBuffer) -> Result<R>
where
    R: Into<VulkanBuffer>,
{
    fn call(&self, _attrs: &OpAttrs, inputs: &[&VulkanBuffer]) -> Result<VulkanBuffer> {
        if inputs.len() < 2 {
            return Err(anyhow!("device kernel expects at least 2 inputs"));
        }
        Ok((self)(inputs[0], inputs[1])?.into())
    }
}

impl<R> DeviceKernelAdapter
    for for<'a, 'b, 'c> fn(&'a VulkanBuffer, &'b VulkanBuffer, &'c VulkanBuffer) -> Result<R>
where
    R: Into<VulkanBuffer>,
{
    fn call(&self, _attrs: &OpAttrs, inputs: &[&VulkanBuffer]) -> Result<VulkanBuffer> {
        if inputs.len() < 3 {
            return Err(anyhow!("device kernel expects at least 3 inputs"));
        }
        Ok((self)(inputs[0], inputs[1], inputs[2])?.into())
    }
}

impl<R> DeviceKernelAdapter
    for for<'a, 'b, 'c, 'd> fn(
        &'a VulkanBuffer,
        &'b VulkanBuffer,
        &'c VulkanBuffer,
        &'d VulkanBuffer,
    ) -> Result<R>
where
    R: Into<VulkanBuffer>,
{
    fn call(&self, _attrs: &OpAttrs, inputs: &[&VulkanBuffer]) -> Result<VulkanBuffer> {
        if inputs.len() < 4 {
            return Err(anyhow!("device kernel expects at least 4 inputs"));
        }
        Ok((self)(inputs[0], inputs[1], inputs[2], inputs[3])?.into())
    }
}

impl<R> DeviceKernelAdapter for for<'a, 'b> fn(&'a OpAttrs, &'b VulkanBuffer) -> Result<R>
where
    R: Into<VulkanBuffer>,
{
    fn call(&self, attrs: &OpAttrs, inputs: &[&VulkanBuffer]) -> Result<VulkanBuffer> {
        let a = inputs
            .get(0)
            .ok_or_else(|| anyhow!("device kernel expects at least 1 input"))?;
        Ok((self)(attrs, a)?.into())
    }
}

impl<R> DeviceKernelAdapter
    for for<'a, 'b, 'c> fn(&'a OpAttrs, &'b VulkanBuffer, &'c VulkanBuffer) -> Result<R>
where
    R: Into<VulkanBuffer>,
{
    fn call(&self, attrs: &OpAttrs, inputs: &[&VulkanBuffer]) -> Result<VulkanBuffer> {
        if inputs.len() < 2 {
            return Err(anyhow!("device kernel expects at least 2 inputs"));
        }
        Ok((self)(attrs, inputs[0], inputs[1])?.into())
    }
}

impl<R> DeviceKernelAdapter
    for for<'a, 'b, 'c, 'd> fn(
        &'a OpAttrs,
        &'b VulkanBuffer,
        &'c VulkanBuffer,
        &'d VulkanBuffer,
    ) -> Result<R>
where
    R: Into<VulkanBuffer>,
{
    fn call(&self, attrs: &OpAttrs, inputs: &[&VulkanBuffer]) -> Result<VulkanBuffer> {
        if inputs.len() < 3 {
            return Err(anyhow!("device kernel expects at least 3 inputs"));
        }
        Ok((self)(attrs, inputs[0], inputs[1], inputs[2])?.into())
    }
}

impl<R> DeviceKernelAdapter
    for for<'a, 'b, 'c, 'd, 'e> fn(
        &'a OpAttrs,
        &'b VulkanBuffer,
        &'c VulkanBuffer,
        &'d VulkanBuffer,
        &'e VulkanBuffer,
    ) -> Result<R>
where
    R: Into<VulkanBuffer>,
{
    fn call(&self, attrs: &OpAttrs, inputs: &[&VulkanBuffer]) -> Result<VulkanBuffer> {
        if inputs.len() < 4 {
            return Err(anyhow!("device kernel expects at least 4 inputs"));
        }
        Ok((self)(attrs, inputs[0], inputs[1], inputs[2], inputs[3])?.into())
    }
}
