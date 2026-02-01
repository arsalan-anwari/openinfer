use anyhow::{anyhow, Result};

use crate::ops::cpu::packed_cpu::{get_bits, sign_extend, PackedBits};
use crate::tensor::{I1, I2, I4, Tensor};

fn is_neg_packed<T: PackedBits>(
    a: &Tensor<T>,
    out: &mut Tensor<bool>,
    width: u8,
) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for idx in 0..a.numel() {
        let value = sign_extend(get_bits(&a.data, idx, width), width);
        out.data[idx] = value < 0;
    }
    Ok(())
}

pub fn is_neg_i1_packed(a: &Tensor<I1>, out: &mut Tensor<bool>) -> Result<()> {
    is_neg_packed(a, out, 1)
}

pub fn is_neg_i2_packed(a: &Tensor<I2>, out: &mut Tensor<bool>) -> Result<()> {
    is_neg_packed(a, out, 2)
}

pub fn is_neg_i4_packed(a: &Tensor<I4>, out: &mut Tensor<bool>) -> Result<()> {
    is_neg_packed(a, out, 4)
}
