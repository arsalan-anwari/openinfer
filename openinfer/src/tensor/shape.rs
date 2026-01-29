use anyhow::{anyhow, Result};

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>()
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1usize;
    for (idx, dim) in shape.iter().rev().enumerate() {
        let i = shape.len() - 1 - idx;
        strides[i] = stride;
        stride = stride.saturating_mul(*dim);
    }
    strides
}

pub(crate) fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    strides == compute_strides(shape)
}

pub(crate) fn offset_for(shape: &[usize], strides: &[usize], indices: &[usize]) -> Result<usize> {
    if shape.len() != indices.len() {
        return Err(anyhow!(
            "expected {} indices, got {}",
            shape.len(),
            indices.len()
        ));
    }
    let mut offset = 0usize;
    for ((dim, stride), idx) in shape.iter().zip(strides.iter()).zip(indices.iter()) {
        if *idx >= *dim {
            return Err(anyhow!("index {} out of bounds for dim {}", idx, dim));
        }
        offset = offset.saturating_add(idx.saturating_mul(*stride));
    }
    Ok(offset)
}

pub(crate) fn view_parts(
    shape: &[usize],
    strides: &[usize],
    indices: &[usize],
) -> Result<(usize, Vec<usize>, Vec<usize>)> {
    if indices.len() > shape.len() {
        return Err(anyhow!(
            "too many indices: got {}, shape has {} dims",
            indices.len(),
            shape.len()
        ));
    }
    let mut offset = 0usize;
    for (idx, (dim, stride)) in indices.iter().zip(shape.iter().zip(strides.iter())) {
        if *idx >= *dim {
            return Err(anyhow!("index {} out of bounds for dim {}", idx, dim));
        }
        offset = offset.saturating_add(idx.saturating_mul(*stride));
    }
    Ok((
        offset,
        shape[indices.len()..].to_vec(),
        strides[indices.len()..].to_vec(),
    ))
}

pub(crate) fn linear_to_indices(linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut rem = linear;
    let mut out = Vec::with_capacity(shape.len());
    let strides = compute_strides(shape);
    for (dim, stride) in shape.iter().zip(strides.iter()) {
        if *stride == 0 {
            out.push(0);
        } else {
            let coord = rem / *stride;
            out.push(coord.min(dim.saturating_sub(1)));
            rem %= *stride;
        }
    }
    out
}
