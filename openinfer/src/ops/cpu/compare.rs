use anyhow::{anyhow, Result};

use crate::tensor::Tensor;

pub fn compare_same_shape<T: Copy>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<bool>,
    mut cmp: impl FnMut(T, T) -> bool,
) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "input shapes {:?} and {:?} must match",
            a.shape(),
            b.shape()
        ));
    }
    if out.shape() != a.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for ((out_slot, lhs), rhs) in out.data.iter_mut().zip(a.data.iter()).zip(b.data.iter()) {
        *out_slot = cmp(*lhs, *rhs);
    }
    Ok(())
}
