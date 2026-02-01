use anyhow::{anyhow, Result};

use crate::ops::cpu::packed_cpu::{get_bits, PackedBits};
use crate::tensor::{I2, I4, U2, U4, Tensor};

fn popcount_bits(value: u8, width: u8) -> u8 {
    let mask = (1u8 << width) - 1;
    (value & mask).count_ones() as u8
}

fn popcount_packed<T: PackedBits>(
    a: &Tensor<T>,
    out: &mut Tensor<u8>,
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
        let value = get_bits(&a.data, idx, width);
        out.data[idx] = popcount_bits(value, width);
    }
    Ok(())
}

pub fn popcount_i2_packed(a: &Tensor<I2>, out: &mut Tensor<u8>) -> Result<()> {
    popcount_packed(a, out, 2)
}

pub fn popcount_i4_packed(a: &Tensor<I4>, out: &mut Tensor<u8>) -> Result<()> {
    popcount_packed(a, out, 4)
}

pub fn popcount_u2_packed(a: &Tensor<U2>, out: &mut Tensor<u8>) -> Result<()> {
    popcount_packed(a, out, 2)
}

pub fn popcount_u4_packed(a: &Tensor<U4>, out: &mut Tensor<u8>) -> Result<()> {
    popcount_packed(a, out, 4)
}
