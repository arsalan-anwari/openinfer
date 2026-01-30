use anyhow::{anyhow, Result};

use crate::ops::cpu::packed_cpu::{get_bits, set_bits, PackedBits};
use crate::tensor::{I1, I2, I4, U1, U2, U4, Tensor};

fn filter_packed<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    cond: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != cond.shape() {
        return Err(anyhow!(
            "input shapes {:?}, {:?}, {:?} must match",
            a.shape(),
            b.shape(),
            cond.shape()
        ));
    }
    if out.shape() != a.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for idx in 0..out.numel() {
        let cond_value = get_bits(&cond.data, idx, width);
        let selected = if cond_value != 0 {
            get_bits(&a.data, idx, width)
        } else {
            get_bits(&b.data, idx, width)
        };
        set_bits(&mut out.data, idx, width, selected);
    }
    Ok(())
}

pub fn filter_i1_packed(a: &Tensor<I1>, b: &Tensor<I1>, cond: &Tensor<I1>, out: &mut Tensor<I1>) -> Result<()> {
    filter_packed(a, b, cond, out, 1)
}

pub fn filter_i2_packed(a: &Tensor<I2>, b: &Tensor<I2>, cond: &Tensor<I2>, out: &mut Tensor<I2>) -> Result<()> {
    filter_packed(a, b, cond, out, 2)
}

pub fn filter_i4_packed(a: &Tensor<I4>, b: &Tensor<I4>, cond: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    filter_packed(a, b, cond, out, 4)
}

pub fn filter_u1_packed(a: &Tensor<U1>, b: &Tensor<U1>, cond: &Tensor<U1>, out: &mut Tensor<U1>) -> Result<()> {
    filter_packed(a, b, cond, out, 1)
}

pub fn filter_u2_packed(a: &Tensor<U2>, b: &Tensor<U2>, cond: &Tensor<U2>, out: &mut Tensor<U2>) -> Result<()> {
    filter_packed(a, b, cond, out, 2)
}

pub fn filter_u4_packed(a: &Tensor<U4>, b: &Tensor<U4>, cond: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    filter_packed(a, b, cond, out, 4)
}
