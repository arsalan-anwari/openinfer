use anyhow::{anyhow, Result};

use crate::graph::OpKind;

pub fn supports_broadcast(op: OpKind) -> bool {
    match op {
        OpKind::Add => super::add::supports_broadcast(),
        OpKind::Mul => super::mul::supports_broadcast(),
        OpKind::Matmul => super::matmul::supports_broadcast(),
        _ => false,
    }
}

#[allow(dead_code)]
pub(crate) fn build_metadata(
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
