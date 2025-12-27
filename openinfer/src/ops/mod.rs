#[cfg(any(
    not(all(target_arch = "x86_64", target_feature = "avx")),
    not(all(target_arch = "x86_64", target_feature = "avx2")),
    not(feature = "vulkan")
))]
mod cpu;

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
mod cpu_avx;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod cpu_avx2;

#[cfg(feature = "vulkan")]
mod vulkan;

mod registry;

pub use cpu::{abs_f32, add_f32, mul_f32};

#[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
pub use cpu_avx::{abs_f32 as abs_f32_avx, add_f32 as add_f32_avx, mul_f32 as mul_f32_avx};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub use cpu_avx2::{abs_f32 as abs_f32_avx2, add_f32 as add_f32_avx2, mul_f32 as mul_f32_avx2};

#[cfg(feature = "vulkan")]
pub use vulkan::{abs_f32 as abs_f32_vulkan, add_f32 as add_f32_vulkan, mul_f32 as mul_f32_vulkan};

pub use registry::{lookup_kernel, KernelFn};
