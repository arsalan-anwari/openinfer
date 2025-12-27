use anyhow::{anyhow, Result};

use crate::backend::VulkanBuffer;
use crate::{OpAttrs, TensorValue};

mod cpu;
#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
mod cpu_avx;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod cpu_avx2;
mod vulkan;
mod registry;

pub use cpu::{add_f32, mul_f32};

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
pub use cpu_avx::{add_f32 as add_f32_avx, mul_f32 as mul_f32_avx};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx")))]
pub fn add_f32_avx(_a: &[f32], _b: &[f32]) -> Result<Vec<f32>> {
    Err(anyhow!("AVX backend not supported for this build"))
}
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx")))]
pub fn mul_f32_avx(_a: &[f32], _b: &[f32]) -> Result<Vec<f32>> {
    Err(anyhow!("AVX backend not supported for this build"))
}

pub fn add_f32_avx_kernel(attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
    host_f32_binary_kernel("add", add_f32_avx, attrs, inputs)
}

pub fn mul_f32_avx_kernel(attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
    host_f32_binary_kernel("mul", mul_f32_avx, attrs, inputs)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use cpu_avx2::{add_f32 as add_f32_avx2, mul_f32 as mul_f32_avx2};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub fn add_f32_avx2(_a: &[f32], _b: &[f32]) -> Result<Vec<f32>> {
    Err(anyhow!("AVX2 backend not supported for this build"))
}
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub fn mul_f32_avx2(_a: &[f32], _b: &[f32]) -> Result<Vec<f32>> {
    Err(anyhow!("AVX2 backend not supported for this build"))
}

pub fn add_f32_avx2_kernel(attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
    host_f32_binary_kernel("add", add_f32_avx2, attrs, inputs)
}

pub fn mul_f32_avx2_kernel(attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
    host_f32_binary_kernel("mul", mul_f32_avx2, attrs, inputs)
}

pub fn add_f32_cpu_kernel(attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
    host_f32_binary_kernel("add", add_f32, attrs, inputs)
}

pub fn mul_f32_cpu_kernel(attrs: &OpAttrs, inputs: &[TensorValue]) -> Result<TensorValue> {
    host_f32_binary_kernel("mul", mul_f32, attrs, inputs)
}

pub use vulkan::{add_f32 as add_f32_vulkan, mul_f32 as mul_f32_vulkan};

pub fn add_f32_vulkan_kernel(attrs: &OpAttrs, inputs: &[&VulkanBuffer]) -> Result<VulkanBuffer> {
    device_f32_binary_kernel("add", add_f32_vulkan, attrs, inputs)
}

pub fn mul_f32_vulkan_kernel(attrs: &OpAttrs, inputs: &[&VulkanBuffer]) -> Result<VulkanBuffer> {
    device_f32_binary_kernel("mul", mul_f32_vulkan, attrs, inputs)
}

pub use registry::{lookup_kernel, KernelFn};

fn host_f32_binary_kernel(
    op: &str,
    func: fn(&[f32], &[f32]) -> Result<Vec<f32>>,
    _attrs: &OpAttrs,
    inputs: &[TensorValue],
) -> Result<TensorValue> {
    let a = inputs
        .get(0)
        .ok_or_else(|| anyhow!("{} op expects at least 2 inputs", op))?
        .as_f32()?;
    let b = inputs
        .get(1)
        .ok_or_else(|| anyhow!("{} op expects at least 2 inputs", op))?
        .as_f32()?;
    let out = func(&a.data, &b.data)?;
    Ok(TensorValue::from(out))
}

fn device_f32_binary_kernel(
    op: &str,
    func: fn(&VulkanBuffer, &VulkanBuffer) -> Result<VulkanBuffer>,
    _attrs: &OpAttrs,
    inputs: &[&VulkanBuffer],
) -> Result<VulkanBuffer> {
    let a = inputs
        .get(0)
        .ok_or_else(|| anyhow!("{} op expects at least 2 inputs", op))?;
    let b = inputs
        .get(1)
        .ok_or_else(|| anyhow!("{} op expects at least 2 inputs", op))?;
    func(a, b)
}
