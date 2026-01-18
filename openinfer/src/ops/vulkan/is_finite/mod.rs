use anyhow::{anyhow, Result};

use crate::backend::vulkan::storage_size_bytes_for_len;
use crate::backend::VulkanBuffer;
use crate::graph::{OpAttrs, OpKind};
use crate::tensor::{compute_strides, DType};
use crate::timer::Timer;

pub mod registry;

pub fn is_finite_generic(
    attrs: &OpAttrs,
    a: &VulkanBuffer,
    thread_id: usize,
) -> Result<VulkanBuffer> {
    let runtime = super::runtime_from_buffers(a, None)?;
    let target = super::spv_target_name(OpKind::IsFinite, a.effective_dtype, attrs)?;
    let entry = "main";
    let spirv = a
        .spv_bytes_for_target(&target)
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for is_finite op", target))?;
    let output_size = storage_size_bytes_for_len(DType::Bool, 1);
    let output_inner = runtime.create_buffer(output_size)?;
    let len = a.len;
    let push = [len as u32, 0, 0, 0];
    let work_len = 1usize;
    let duration_ns = runtime.dispatch(
        OpKind::IsFinite,
        a.effective_dtype,
        &target,
        entry,
        spirv,
        &a.inner,
        &a.inner,
        &output_inner,
        push,
        work_len,
    )?;
    Timer::record(thread_id, duration_ns);
    let shape = Vec::new();
    let strides = compute_strides(shape.as_slice());
    Ok(VulkanBuffer {
        dtype: DType::Bool,
        effective_dtype: DType::Bool,
        len: 1,
        shape,
        strides,
        shader: a.shader.clone(),
        inner: output_inner,
    })
}

pub(crate) fn spv_target_name_is_finite(dtype: DType, attrs: &OpAttrs) -> Result<String> {
    match (dtype, attrs) {
        (DType::I8, &OpAttrs::None)
        | (DType::I16, &OpAttrs::None)
        | (DType::F32, &OpAttrs::None)
        | (DType::Bool, &OpAttrs::None)
        | (DType::U8, &OpAttrs::None)
        | (DType::U16, &OpAttrs::None)
        | (DType::I32, &OpAttrs::None)
        | (DType::U32, &OpAttrs::None)
        | (DType::I64, &OpAttrs::None)
        | (DType::U64, &OpAttrs::None) => {
            Ok(format!(
                "is_finite_scalar_{}",
                super::dtype_suffix(dtype).unwrap()
            ))
        }
        _ => Err(anyhow!(
            "no Vulkan SPIR-V target for is_finite dtype {:?}, attrs {:?}",
            dtype,
            attrs
        )),
    }
}
