use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs};
use crate::tensor::compute_strides;

pub fn axes_from_attrs(attrs: &OpAttrs, rank: usize) -> Result<Vec<usize>> {
    let axes = attrs
        .items
        .iter()
        .find(|attr| attr.name == "axes")
        .ok_or_else(|| anyhow!("missing axes attribute"))?;
    let raw = match &axes.value {
        AttrValue::IntList(values) => values.clone(),
        _ => return Err(anyhow!("axes attribute must be int list")),
    };
    normalize_axes(&raw, rank)
}

pub fn axis_from_attrs(attrs: &OpAttrs, rank: usize) -> Result<usize> {
    let axis = attrs
        .items
        .iter()
        .find(|attr| attr.name == "axis")
        .ok_or_else(|| anyhow!("missing axis attribute"))?;
    let raw = match &axis.value {
        AttrValue::Int(value) => *value,
        AttrValue::UInt(value) => *value as i64,
        AttrValue::Bool(value) => if *value { 1 } else { 0 },
        _ => return Err(anyhow!("axis attribute must be int")),
    };
    normalize_axis(raw, rank)
}

pub fn keepdims_from_attrs(attrs: &OpAttrs) -> bool {
    attrs
        .items
        .iter()
        .find(|attr| attr.name == "keepdims")
        .and_then(|attr| match &attr.value {
            AttrValue::Bool(value) => Some(*value),
            AttrValue::Int(value) => Some(*value != 0),
            AttrValue::UInt(value) => Some(*value != 0),
            _ => None,
        })
        .unwrap_or(false)
}

pub fn select_first_from_attrs(attrs: &OpAttrs) -> bool {
    attrs
        .items
        .iter()
        .find(|attr| attr.name == "select_first")
        .and_then(|attr| match &attr.value {
            AttrValue::Bool(value) => Some(*value),
            AttrValue::Int(value) => Some(*value != 0),
            AttrValue::UInt(value) => Some(*value != 0),
            _ => None,
        })
        .unwrap_or(true)
}

pub fn output_shape(shape: &[usize], axes: &[usize], keepdims: bool) -> Vec<usize> {
    if keepdims {
        shape
            .iter()
            .enumerate()
            .map(|(idx, dim)| if axes.contains(&idx) { 1 } else { *dim })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter_map(|(idx, dim)| if axes.contains(&idx) { None } else { Some(*dim) })
            .collect()
    }
}

pub fn output_offset(
    input_indices: &[usize],
    axes: &[usize],
    keepdims: bool,
    out_strides: &[usize],
) -> usize {
    let mut coords = Vec::with_capacity(out_strides.len());
    for (idx, coord) in input_indices.iter().enumerate() {
        if axes.contains(&idx) {
            if keepdims {
                coords.push(0);
            }
        } else {
            coords.push(*coord);
        }
    }
    coords
        .iter()
        .zip(out_strides.iter())
        .map(|(i, s)| i.saturating_mul(*s))
        .sum()
}

pub fn reduce_count(shape: &[usize], axes: &[usize]) -> usize {
    axes
        .iter()
        .map(|axis| shape.get(*axis).copied().unwrap_or(1))
        .product()
}

pub fn output_strides(shape: &[usize]) -> Vec<usize> {
    compute_strides(shape)
}

pub fn linear_to_indices(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let strides = compute_strides(shape);
    let mut indices = Vec::with_capacity(shape.len());
    for (dim, stride) in shape.iter().zip(strides.iter()) {
        if *stride == 0 {
            indices.push(0);
            continue;
        }
        let coord = linear / *stride;
        indices.push(coord.min(dim.saturating_sub(1)));
        linear %= *stride;
    }
    indices
}

fn normalize_axes(raw: &[i64], rank: usize) -> Result<Vec<usize>> {
    if raw.is_empty() {
        return Err(anyhow!("axes attribute must not be empty"));
    }
    let mut axes = Vec::with_capacity(raw.len());
    for axis in raw {
        axes.push(normalize_axis(*axis, rank)?);
    }
    axes.sort_unstable();
    axes.dedup();
    Ok(axes)
}

fn normalize_axis(axis: i64, rank: usize) -> Result<usize> {
    let rank_i64 = rank as i64;
    let resolved = if axis < 0 { axis + rank_i64 } else { axis };
    if resolved < 0 || resolved >= rank_i64 {
        return Err(anyhow!("axis {} out of range for rank {}", axis, rank));
    }
    Ok(resolved as usize)
}
