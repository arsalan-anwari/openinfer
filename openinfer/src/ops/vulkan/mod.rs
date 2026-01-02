pub mod abs;
pub mod add;
pub mod mul;
pub mod registry;

use crate::backend::VulkanBuffer;
use anyhow::{anyhow, Result};
use std::sync::Arc;

use crate::backend::vulkan::VulkanRuntime;
use crate::tensor::DType;

pub(crate) fn runtime_from_buffers(
    a: &VulkanBuffer,
    b: Option<&VulkanBuffer>,
) -> Result<Arc<VulkanRuntime>> {
    let runtime = Arc::clone(a.inner.runtime());
    if let Some(b) = b {
        if !Arc::ptr_eq(a.inner.runtime(), b.inner.runtime()) {
            return Err(anyhow!("vulkan buffers are from different runtimes"));
        }
    }
    Ok(runtime)
}

pub(crate) fn entry_point_name(op: &str, dtype: DType) -> Result<&'static str> {
    let supported = match op {
        "abs" => matches!(dtype, DType::I8 | DType::I16 | DType::F32 | DType::I32 | DType::I64),
        "add" | "mul" => matches!(
            dtype,
            DType::I8
                | DType::I16
                | DType::F32
                | DType::Bool
                | DType::U8
                | DType::U16
                | DType::I32
                | DType::U32
                | DType::I64
                | DType::U64
        ),
        _ => false,
    };
    if supported {
        Ok("main")
    } else {
        Err(anyhow!("no Vulkan entry point for dtype {:?}", dtype))
    }
}
