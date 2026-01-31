use anyhow::{anyhow, Result};

use crate::ops::cpu::packed_cpu::{get_bits, sign_extend, PackedBits};
use crate::tensor::Tensor;

pub fn cast_packed_signed<TIn: PackedBits, TOut: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<TOut>,
    width: u8,
    mut f: impl FnMut(i8) -> TOut,
) -> Result<()> {
    if input.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            input.shape()
        ));
    }
    let len = input.numel();
    for idx in 0..len {
        let raw = get_bits(&input.data, idx, width);
        let value = sign_extend(raw, width);
        out.data[idx] = f(value);
    }
    Ok(())
}

pub fn cast_packed_unsigned<TIn: PackedBits, TOut: Copy>(
    input: &Tensor<TIn>,
    out: &mut Tensor<TOut>,
    width: u8,
    mut f: impl FnMut(u8) -> TOut,
) -> Result<()> {
    if input.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            input.shape()
        ));
    }
    let len = input.numel();
    for idx in 0..len {
        let raw = get_bits(&input.data, idx, width);
        out.data[idx] = f(raw);
    }
    Ok(())
}
