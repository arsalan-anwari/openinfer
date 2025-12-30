use anyhow::Result;

use crate::backend::VulkanBuffer;
use crate::tensor::DType;

pub mod registry;

pub fn abs_generic(a: &VulkanBuffer) -> Result<VulkanBuffer> {
    if matches!(a.dtype, DType::Bitset | DType::F16) {
        println!("vulkan abs stub: dtype={:?}", a.dtype);
        return Ok(VulkanBuffer { dtype: a.dtype, len: 0 });
    }
    println!("vulkan abs: dtype={:?} len={}", a.dtype, a.len);
    Ok(VulkanBuffer {
        dtype: a.dtype,
        len: a.len,
    })
}
