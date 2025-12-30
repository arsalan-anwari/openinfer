use anyhow::{anyhow, Result};

use crate::backend::VulkanBuffer;
use crate::tensor::DType;

pub mod registry;

pub fn mul_generic(a: &VulkanBuffer, b: &VulkanBuffer) -> Result<VulkanBuffer> {
    if a.len != b.len {
        return Err(anyhow!("mul op shape mismatch"));
    }
    if a.dtype != b.dtype {
        return Err(anyhow!("mul op expects matching dtypes"));
    }
    if matches!(a.dtype, DType::Bitset | DType::F16) {
        println!("vulkan mul stub: dtype={:?}", a.dtype);
        return Ok(VulkanBuffer { dtype: a.dtype, len: 0 });
    }
    println!("vulkan mul: dtype={:?} len={}", a.dtype, a.len);
    Ok(VulkanBuffer {
        dtype: a.dtype,
        len: a.len,
    })
}
