use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::ops::cpu::packed_cpu::{get_bits, set_bits, sign_extend, PackedBits};
use crate::tensor::{I4, Tensor};

use super::common::relu_params_i64;

fn relu_packed_signed<T: PackedBits>(
    attrs: &OpAttrs,
    a: &Tensor<T>,
    out: &mut Tensor<T>,
    width: u8,
) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    let (alpha, clamp_max) = relu_params_i64(attrs)?;
    let total = out.numel();
    for idx in 0..total {
        let value = sign_extend(get_bits(&a.data, idx, width), width) as i64;
        let mut y = if value >= 0 { value } else { value.wrapping_mul(alpha) };
        if y > clamp_max {
            y = clamp_max;
        }
        set_bits(&mut out.data, idx, width, y as u8);
    }
    Ok(())
}

fn relu_packed_signed_inplace<T: PackedBits>(attrs: &OpAttrs, a: &mut Tensor<T>, width: u8) -> Result<()> {
    let (alpha, clamp_max) = relu_params_i64(attrs)?;
    let total = a.numel();
    for idx in 0..total {
        let value = sign_extend(get_bits(&a.data, idx, width), width) as i64;
        let mut y = if value >= 0 { value } else { value.wrapping_mul(alpha) };
        if y > clamp_max {
            y = clamp_max;
        }
        set_bits(&mut a.data, idx, width, y as u8);
    }
    Ok(())
}

pub fn relu_i4_packed(attrs: &OpAttrs, a: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    relu_packed_signed(attrs, a, out, 4)
}

pub fn relu_i4_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<I4>) -> Result<()> {
    relu_packed_signed_inplace(attrs, a, 4)
}
