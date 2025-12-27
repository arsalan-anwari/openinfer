use anyhow::Result;

use crate::backend::VulkanBuffer;
use crate::graph::{OpAttrs, OpKind};
use crate::tensor::{DType, TensorValue};
use crate::executor::Device;

use super::{
    add_f32_avx2_kernel, add_f32_avx_kernel, add_f32_cpu_kernel, add_f32_vulkan_kernel,
    mul_f32_avx2_kernel, mul_f32_avx_kernel, mul_f32_cpu_kernel, mul_f32_vulkan_kernel,
};

pub type HostKernel = fn(&OpAttrs, &[TensorValue]) -> Result<TensorValue>;
pub type VulkanKernel = fn(&OpAttrs, &[&VulkanBuffer]) -> Result<VulkanBuffer>;

pub enum KernelFn {
    Host(HostKernel),
    Vulkan(VulkanKernel),
}

pub fn lookup_kernel(
    device: Device,
    op: OpKind,
    dtype: DType,
    attrs: OpAttrs,
) -> Option<KernelFn> {
    match (device, op, dtype, attrs) {
        (Device::Cpu, OpKind::Add, DType::F32, OpAttrs::None) => {
            Some(KernelFn::Host(add_f32_cpu_kernel))
        }
        (Device::Cpu, OpKind::Mul, DType::F32, OpAttrs::None) => {
            Some(KernelFn::Host(mul_f32_cpu_kernel))
        }
        (Device::CpuAvx, OpKind::Add, DType::F32, OpAttrs::None) => {
            Some(KernelFn::Host(add_f32_avx_kernel))
        }
        (Device::CpuAvx, OpKind::Mul, DType::F32, OpAttrs::None) => {
            Some(KernelFn::Host(mul_f32_avx_kernel))
        }
        (Device::CpuAvx2, OpKind::Add, DType::F32, OpAttrs::None) => {
            Some(KernelFn::Host(add_f32_avx2_kernel))
        }
        (Device::CpuAvx2, OpKind::Mul, DType::F32, OpAttrs::None) => {
            Some(KernelFn::Host(mul_f32_avx2_kernel))
        }
        (Device::Vulkan, OpKind::Add, DType::F32, OpAttrs::None) => {
            Some(KernelFn::Vulkan(add_f32_vulkan_kernel))
        }
        (Device::Vulkan, OpKind::Mul, DType::F32, OpAttrs::None) => {
            Some(KernelFn::Vulkan(mul_f32_vulkan_kernel))
        }
        _ => None,
    }
}
