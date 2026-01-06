use anyhow::Result;

use crate::executor::Device;
use crate::graph::{OpAttrs, OpKind};
use crate::tensor::{DType, TensorValue};

pub type HostKernel =
    Box<dyn Fn(&OpAttrs, &[TensorValue], u32) -> Result<TensorValue> + Send + Sync>;
#[cfg(feature = "vulkan")]
pub type VulkanKernel =
    Box<dyn Fn(&OpAttrs, &[&crate::backend::VulkanBuffer], u32) -> Result<crate::backend::VulkanBuffer> + Send + Sync>;

pub enum KernelFn {
    Host(HostKernel),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanKernel),
}

pub fn lookup_kernel(
    device: Device,
    op: OpKind,
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: OpAttrs,
) -> Option<KernelFn> {
    match device {
        Device::Cpu => super::cpu::registry::lookup_kernel_cpu(
            op,
            output_dtype,
            input_dtypes,
            attrs,
        ),
        #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
        Device::CpuAvx => super::cpu_avx::registry::lookup_kernel_cpu_avx(
            op,
            output_dtype,
            input_dtypes,
            attrs,
        ),
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
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
