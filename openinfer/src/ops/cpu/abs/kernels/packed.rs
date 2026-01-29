use anyhow::{anyhow, Result};

use crate::ops::cpu::packed_cpu::{get_bits, set_bits, sign_extend, PackedBits};
use crate::tensor::{I1, I2, I4, Tensor};

fn abs_packed_signed<T: PackedBits>(a: &Tensor<T>, out: &mut Tensor<T>, width: u8) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for idx in 0..out.len() {
        let value = sign_extend(get_bits(&a.data, idx, width), width);
        let abs = value.wrapping_abs() as u8;
        set_bits(&mut out.data, idx, width, abs);
    }
    Ok(())
}

fn abs_packed_signed_inplace<T: PackedBits>(a: &mut Tensor<T>, width: u8) -> Result<()> {
    for idx in 0..a.len() {
        let value = sign_extend(get_bits(&a.data, idx, width), width);
        let abs = value.wrapping_abs() as u8;
        set_bits(&mut a.data, idx, width, abs);
    }
    Ok(())
}

pub fn abs_i1_packed(a: &Tensor<I1>, out: &mut Tensor<I1>) -> Result<()> {
    abs_packed_signed(a, out, 1)
}

pub fn abs_i2_packed(a: &Tensor<I2>, out: &mut Tensor<I2>) -> Result<()> {
    abs_packed_signed(a, out, 2)
}

pub fn abs_i4_packed(a: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    abs_packed_signed(a, out, 4)
}

pub fn abs_i1_packed_inplace(a: &mut Tensor<I1>) -> Result<()> {
    abs_packed_signed_inplace(a, 1)
}

pub fn abs_i2_packed_inplace(a: &mut Tensor<I2>) -> Result<()> {
    abs_packed_signed_inplace(a, 2)
}

pub fn abs_i4_packed_inplace(a: &mut Tensor<I4>) -> Result<()> {
    abs_packed_signed_inplace(a, 4)
}
