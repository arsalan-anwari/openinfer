pub mod abs;
pub mod add;
pub mod mul;
pub mod relu;
pub mod registry;

use crate::backend::VulkanBuffer;
use anyhow::{anyhow, Result};
use std::sync::Arc;

use crate::backend::vulkan::VulkanRuntime;
use crate::graph::OpAttrs;
use crate::graph::OpKind;
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

pub(crate) fn entry_point_name() -> &'static str {
    "main"
}

pub(crate) fn spv_target_name(op: OpKind, dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match op {
        OpKind::Abs => abs::spv_target_name_abs(dtype, attrs),
        OpKind::Add => add::spv_target_name_add(dtype, attrs),
        OpKind::Mul => mul::spv_target_name_mul(dtype, attrs),
        OpKind::Relu => relu::spv_target_name_relu(dtype, attrs),
    }
}

pub(crate) fn dtype_suffix(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::I8 => Some("i8"),
        DType::I16 => Some("i16"),
        DType::F32 => Some("f32"),
        DType::Bool => Some("bool"),
        DType::U8 => Some("u8"),
        DType::U16 => Some("u16"),
        DType::I32 => Some("i32"),
        DType::U32 => Some("u32"),
        DType::I64 => Some("i64"),
        DType::U64 => Some("u64"),
        _ => None,
    }
}
