pub(crate) mod cpu;

#[cfg(feature = "avx")]
pub(crate) mod cpu_avx;

#[cfg(feature = "avx2")]
pub(crate) mod cpu_avx2;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan;

pub(crate) mod registry;
mod adapter;
#[allow(unused_imports)]
pub use registry::{
    broadcast_enabled,
    broadcast_is_elementwise,
    broadcast_policy,
    broadcast_requires_materialize,
    lookup_kernel,
    lookup_kernel_inplace,
    BroadcastPolicy,
    InplaceKernelFn,
    KernelFn,
};

#[allow(unused_imports)]
pub use adapter::{cpu_kernel, CpuKernelAdapter};
#[cfg(feature = "vulkan")]
#[allow(unused_imports)]
pub use adapter::{device_kernel, DeviceKernelAdapter};
