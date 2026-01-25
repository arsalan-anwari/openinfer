use anyhow::{anyhow, Result};

use crate::tensor::{broadcast_shapes, broadcast_strides, compute_strides, numel, Tensor};
use crate::timer::Timer;

pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    strides == compute_strides(shape)
}

pub fn ensure_same_shape<T, O>(a: &Tensor<T>, b: &Tensor<T>, out: &Tensor<O>) -> Result<()> {
    if a.shape() != b.shape() || out.shape() != a.shape() {
        return Err(anyhow!(
            "shape mismatch: a {:?}, b {:?}, out {:?}",
            a.shape(),
            b.shape(),
            out.shape()
        ));
    }
    Ok(())
}

pub fn ensure_same_len<T, O>(a: &Tensor<T>, b: &Tensor<T>, out: &Tensor<O>) -> Result<()> {
    if a.data.len() != b.data.len() || out.data.len() != a.data.len() {
        return Err(anyhow!("data length mismatch"));
    }
    Ok(())
}

pub fn ensure_same_shape_unary<T, O>(a: &Tensor<T>, out: &Tensor<O>) -> Result<()> {
    if out.shape() != a.shape() {
        return Err(anyhow!(
            "shape mismatch: input {:?}, out {:?}",
            a.shape(),
            out.shape()
        ));
    }
    Ok(())
}

pub fn ensure_same_len_unary<T, O>(a: &Tensor<T>, out: &Tensor<O>) -> Result<()> {
    if out.data.len() != a.data.len() {
        return Err(anyhow!("data length mismatch"));
    }
    Ok(())
}

pub fn broadcast_binary<T, F>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<T>,
    mut f: F,
    thread_id: usize,
) -> Result<()>
where
    T: Clone,
    F: FnMut(&T, &T) -> T,
{
    let expected = broadcast_shapes(a.shape(), b.shape())?;
    if expected != out.shape() {
        return Err(anyhow!(
            "broadcast output shape mismatch: expected {:?}, got {:?}",
            expected,
            out.shape()
        ));
    }
    if !is_contiguous(out.shape(), out.strides()) {
        return Err(anyhow!("broadcast output must be contiguous"));
    }
    let out_len = numel(out.shape());
    if out.data.len() != out_len {
        return Err(anyhow!(
            "broadcast output len mismatch: expected {}, got {}",
            out_len,
            out.data.len()
        ));
    }
    if out_len == 0 {
        return Ok(());
    }
    let a_strides = broadcast_strides(a.shape(), a.strides(), out.shape())?;
    let b_strides = broadcast_strides(b.shape(), b.strides(), out.shape())?;
    let mut coords = vec![0usize; out.shape().len()];
    Timer::start(thread_id);
    for idx in 0..out_len {
        let mut a_offset = 0usize;
        let mut b_offset = 0usize;
        for (dim, coord) in coords.iter().enumerate() {
            a_offset = a_offset.saturating_add(coord.saturating_mul(a_strides[dim]));
            b_offset = b_offset.saturating_add(coord.saturating_mul(b_strides[dim]));
        }
        out.data[idx] = f(&a.data[a_offset], &b.data[b_offset]);
        for dim in (0..coords.len()).rev() {
            coords[dim] += 1;
            if coords[dim] < out.shape()[dim] {
                break;
            }
            coords[dim] = 0;
        }
    }
    Timer::stop(thread_id);
    Ok(())
}
