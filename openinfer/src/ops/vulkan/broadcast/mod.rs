use anyhow::{anyhow, Result};

use crate::backend::vulkan::{embedded_spirv_for_op, storage_size_bytes};
use crate::backend::VulkanBuffer;
use crate::graph::OpKind;
use crate::tensor::{compute_strides, numel, DType};

pub fn broadcast_buffer(
    input: &VulkanBuffer,
    out_shape: &[usize],
    _thread_id: usize,
) -> Result<VulkanBuffer> {
    let runtime = super::runtime_from_buffers(input, None)?;
    let len = numel(out_shape);
    let target = spv_target_name_broadcast(input.dtype)?;
    let entry = "main";
    let spirv_map = embedded_spirv_for_op("broadcast");
    let spirv = spirv_map
        .get(&target)
        .copied()
        .ok_or_else(|| anyhow!("missing SPIR-V target {} for broadcast", target))?;

    let meta = build_metadata(input.shape.as_slice(), input.strides.as_slice(), out_shape)?;
    let meta_bytes: Vec<u8> = meta.iter().flat_map(|v| v.to_le_bytes()).collect();
    let meta_inner = runtime.create_buffer(meta_bytes.len())?;
    runtime.write_buffer(&meta_inner, &meta_bytes)?;

    let output_size = storage_size_bytes(input.dtype) * len;
    let output_inner = runtime.create_buffer(output_size)?;
    let push = [len as u32, 0, 0, 0];
    let _duration_ns = runtime.dispatch(
        // Reuse a stable op key for pipeline caching; the target string keeps it distinct.
        OpKind::Add,
        input.dtype,
        &target,
        entry,
        spirv,
        &input.inner,
        &meta_inner,
        &output_inner,
        push,
        len,
    )?;

    Ok(VulkanBuffer {
        dtype: input.dtype,
        len,
        shape: out_shape.to_vec(),
        strides: compute_strides(out_shape),
        shader: input.shader.clone(),
        inner: output_inner,
    })
}

fn spv_target_name_broadcast(dtype: DType) -> Result<String> {
    let suffix = super::dtype_suffix(dtype)
        .ok_or_else(|| anyhow!("broadcast not supported for dtype {:?}", dtype))?;
    Ok(format!("broadcast_{}", suffix))
}

fn build_metadata(
    in_shape: &[usize],
    in_strides: &[usize],
    out_shape: &[usize],
) -> Result<Vec<u32>> {
    if in_shape.len() != in_strides.len() {
        return Err(anyhow!(
            "broadcast metadata expects shape/stride rank match, got {} and {}",
            in_shape.len(),
            in_strides.len()
        ));
    }
    let rank_out = out_shape.len();
    let rank_in = in_shape.len();
    let mut aligned_shape = vec![1usize; rank_out.saturating_sub(rank_in)];
    aligned_shape.extend_from_slice(in_shape);
    let mut aligned_strides = vec![0usize; rank_out.saturating_sub(rank_in)];
    aligned_strides.extend_from_slice(in_strides);

    let mut meta = Vec::with_capacity(2 + rank_out * 3);
    meta.push(u32::try_from(rank_out).map_err(|_| anyhow!("rank out overflow"))?);
    meta.push(u32::try_from(rank_in).map_err(|_| anyhow!("rank in overflow"))?);
    for dim in out_shape {
        meta.push(u32::try_from(*dim).map_err(|_| anyhow!("shape overflow"))?);
    }
    for dim in &aligned_shape {
        meta.push(u32::try_from(*dim).map_err(|_| anyhow!("shape overflow"))?);
    }
    for stride in &aligned_strides {
        meta.push(u32::try_from(*stride).map_err(|_| anyhow!("stride overflow"))?);
    }
    Ok(meta)
}
