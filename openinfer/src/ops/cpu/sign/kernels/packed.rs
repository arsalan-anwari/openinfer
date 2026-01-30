use anyhow::{anyhow, Result};

use crate::ops::cpu::broadcast::{broadcast_strides, for_each_broadcast_index};
use crate::ops::cpu::packed_cpu::{get_bits, sign_extend, PackedBits};
use crate::tensor::{I1, I2, I4, Tensor};

fn sign_packed<T: PackedBits>(a: &Tensor<T>, out: &mut Tensor<i8>, width: u8) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    let out_shape = out.shape().to_vec();
    let out_strides = out.strides().to_vec();
    let a_strides = broadcast_strides(a.shape(), a.strides(), out_shape.len());
    let zeros = vec![0; out_shape.len()];
    for_each_broadcast_index(
        &out_shape,
        &out_strides,
        &a_strides,
        &zeros,
        |out_offset, a_offset, _| {
            let value = sign_extend(get_bits(&a.data, a_offset, width), width);
            out.data[out_offset] = if value > 0 { 1 } else if value < 0 { -1 } else { 0 };
        },
    );
    Ok(())
}

pub fn sign_i1_packed(a: &Tensor<I1>, out: &mut Tensor<i8>) -> Result<()> {
    sign_packed(a, out, 1)
}

pub fn sign_i2_packed(a: &Tensor<I2>, out: &mut Tensor<i8>) -> Result<()> {
    sign_packed(a, out, 2)
}

pub fn sign_i4_packed(a: &Tensor<I4>, out: &mut Tensor<i8>) -> Result<()> {
    sign_packed(a, out, 4)
}
