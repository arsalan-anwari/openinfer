use anyhow::{anyhow, Result};

use crate::backend::VulkanBuffer;
use crate::tensor::DType;

pub fn add_f32(a: &VulkanBuffer, b: &VulkanBuffer) -> Result<VulkanBuffer> {
    if a.len != b.len {
        return Err(anyhow!("add op shape mismatch"));
    }
    if a.dtype != DType::F32 || b.dtype != DType::F32 {
        return Err(anyhow!("add op expects f32 buffers"));
    }
    println!("vulkan add_f32: len={}", a.len);
    Ok(VulkanBuffer {
        dtype: DType::F32,
        len: a.len,
    })
}

pub fn mul_f32(a: &VulkanBuffer, b: &VulkanBuffer) -> Result<VulkanBuffer> {
    if a.len != b.len {
        return Err(anyhow!("mul op shape mismatch"));
    }
    if a.dtype != DType::F32 || b.dtype != DType::F32 {
        return Err(anyhow!("mul op expects f32 buffers"));
    }
    println!("vulkan mul_f32: len={}", a.len);
    Ok(VulkanBuffer {
        dtype: DType::F32,
        len: a.len,
    })
}

pub fn abs_f32(a: &VulkanBuffer) -> Result<VulkanBuffer> {
    if a.dtype != DType::F32 {
        return Err(anyhow!("abs op expects f32 buffer"));
    }
    println!("vulkan abs_f32: len={}", a.len);
    Ok(VulkanBuffer {
        dtype: DType::F32,
        len: a.len,
    })
}
