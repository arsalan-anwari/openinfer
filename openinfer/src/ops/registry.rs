use anyhow::Result;

use crate::simulator::Device;
use crate::graph::{OpAttrs, OpKind};
use crate::tensor::{DType, TensorValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BroadcastPolicy {
    None,
    CpuOnly,
    AllDevices,
}

pub fn broadcast_policy(op: OpKind) -> BroadcastPolicy {
    // Central switch: add ops here to enable broadcasting across devices.
    match op {
        OpKind::Add | OpKind::Mul => BroadcastPolicy::AllDevices,
        _ => BroadcastPolicy::None,
    }
}

pub fn broadcast_enabled(op: OpKind, device: Device) -> bool {
    match broadcast_policy(op) {
        BroadcastPolicy::None => false,
        BroadcastPolicy::CpuOnly => matches!(device, Device::Cpu | Device::CpuAvx | Device::CpuAvx2),
        BroadcastPolicy::AllDevices => true,
    }
}

pub type HostKernel = Box<
    dyn Fn(&OpAttrs, &[TensorValue], Option<&mut TensorValue>, usize) -> Result<Option<TensorValue>>
        + Send
        + Sync,
>;
#[cfg(feature = "vulkan")]
pub type VulkanKernel =
    Box<dyn Fn(&OpAttrs, &[&crate::backend::VulkanBuffer], usize) -> Result<crate::backend::VulkanBuffer> + Send + Sync>;

pub enum KernelFn {
    Host(HostKernel),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanKernel),
}

pub type HostInplaceKernel =
    Box<dyn Fn(&OpAttrs, &mut TensorValue, &[TensorValue], usize) -> Result<()> + Send + Sync>;
#[cfg(feature = "vulkan")]
pub type VulkanInplaceKernel = VulkanKernel;

pub enum InplaceKernelFn {
    Host(HostInplaceKernel),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanInplaceKernel),
}

pub fn lookup_kernel(
    device: Device,
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match device {
        Device::Cpu => super::cpu::registry::lookup_kernel_cpu(
            op,
            output_dtype,
            input_dtypes,
            attrs,
        ),
        #[cfg(feature = "avx")]
        Device::CpuAvx => super::cpu_avx::registry::lookup_kernel_cpu_avx(
            op,
            output_dtype,
            input_dtypes,
            attrs,
        ),
        #[cfg(feature = "avx2")]
        Device::CpuAvx2 => super::cpu_avx2::registry::lookup_kernel_cpu_avx2(
            op,
            output_dtype,
            input_dtypes,
            attrs,
        ),
        #[cfg(feature = "vulkan")]
        Device::Vulkan => super::vulkan::registry::lookup_kernel_vulkan(
            op,
            output_dtype,
            input_dtypes,
            attrs,
        ),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}

pub fn lookup_kernel_inplace(
    device: Device,
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<InplaceKernelFn> {
    match device {
        Device::Cpu => super::cpu::registry_inplace::lookup_kernel_cpu_inplace(
            op,
            output_dtype,
            input_dtypes,
            attrs,
        ),
        #[cfg(feature = "avx")]
        Device::CpuAvx => super::cpu_avx::registry_inplace::lookup_kernel_cpu_avx_inplace(
            op,
            output_dtype,
            input_dtypes,
            attrs,
        ),
        #[cfg(feature = "avx2")]
        Device::CpuAvx2 => super::cpu_avx2::registry_inplace::lookup_kernel_cpu_avx2_inplace(
            op,
            output_dtype,
            input_dtypes,
            attrs,
        ),
        #[cfg(feature = "vulkan")]
        Device::Vulkan => super::vulkan::registry::lookup_kernel_vulkan_inplace(
            op,
            output_dtype,
            input_dtypes,
            attrs,
        )
        .and_then(|kernel| match kernel {
            KernelFn::Vulkan(func) => Some(InplaceKernelFn::Vulkan(func)),
            KernelFn::Host(_) => None,
        }),
        #[allow(unreachable_patterns)]
        _ => None,
    }
}
