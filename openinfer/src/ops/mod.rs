mod cpu;

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
mod cpu_avx;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod cpu_avx2;

#[cfg(feature = "vulkan")]
mod vulkan;

mod registry;
mod adapter;

pub use registry::{lookup_kernel, KernelFn};

#[allow(unused_imports)]
pub use adapter::{cpu_kernel, CpuKernelAdapter};
#[cfg(feature = "vulkan")]
#[allow(unused_imports)]
pub use adapter::{device_kernel, DeviceKernelAdapter};
