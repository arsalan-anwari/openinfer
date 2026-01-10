mod cpu;

#[cfg(feature = "avx")]
mod cpu_avx;

#[cfg(feature = "avx2")]
mod cpu_avx2;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan;

mod registry;
mod adapter;

#[allow(unused_imports)]
pub use registry::{broadcast_enabled, broadcast_policy, BroadcastPolicy, lookup_kernel, KernelFn};

#[allow(unused_imports)]
pub use adapter::{cpu_kernel, CpuKernelAdapter};
#[cfg(feature = "vulkan")]
#[allow(unused_imports)]
pub use adapter::{device_kernel, DeviceKernelAdapter};
