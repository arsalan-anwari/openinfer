#[cfg(any(
    not(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx")),
    not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))
))]
use anyhow::{anyhow, Result};

#[cfg(any(
    not(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx")),
    not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))
))]
use crate::Tensor;

mod cpu;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx"))]
mod cpu_avx;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
mod cpu_avx2;
mod vulkan;

pub use cpu::{add_f32, mul_f32};

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx"))]
pub use cpu_avx::{add_f32 as add_f32_avx, mul_f32 as mul_f32_avx};
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx")))]
pub fn add_f32_avx(_a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>> {
    Err(anyhow!("AVX backend not supported for this build"))
}
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx")))]
pub fn mul_f32_avx(_a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>> {
    Err(anyhow!("AVX backend not supported for this build"))
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
pub use cpu_avx2::{add_f32 as add_f32_avx2, mul_f32 as mul_f32_avx2};
#[cfg(not(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
)))]
pub fn add_f32_avx2(_a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>> {
    Err(anyhow!("AVX2 backend not supported for this build"))
}
#[cfg(not(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
)))]
pub fn mul_f32_avx2(_a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>> {
    Err(anyhow!("AVX2 backend not supported for this build"))
}

pub use vulkan::{add_f32 as add_f32_vulkan, mul_f32 as mul_f32_vulkan};
