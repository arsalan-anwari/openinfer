use anyhow::{anyhow, Result};

#[allow(unused_imports)]
use crate::tensor::{
    Bitset, BF16, F16, F8E5M2, I1, I2, I4, Tensor, TensorValue, U1, U2, U4,
};

use super::accumulate::{add_accumulate_signed, add_accumulate_unsigned};
use super::common::{SignedInput, UnsignedInput};
use super::normal::{add_inplace, add_normal};
use super::packed::{
    add_packed_signed, add_packed_signed_inplace, add_packed_unsigned, add_packed_unsigned_inplace,
};

fn expect_output(output: Option<&mut TensorValue>) -> Result<&mut TensorValue> {
    output.ok_or_else(|| anyhow!("missing output tensor"))
}

fn add_accumulate_signed_out<In: SignedInput>(
    a: &Tensor<In>,
    b: &Tensor<In>,
    out: &mut TensorValue,
) -> Result<()> {
    match out {
        TensorValue::I8(out) => add_accumulate_signed::<In, i8>(a, b, out),
        TensorValue::I16(out) => add_accumulate_signed::<In, i16>(a, b, out),
        TensorValue::I32(out) => add_accumulate_signed::<In, i32>(a, b, out),
        TensorValue::I64(out) => add_accumulate_signed::<In, i64>(a, b, out),
        _ => Err(anyhow!("dtype mismatch")),
    }
}

fn add_accumulate_unsigned_out<In: UnsignedInput>(
    a: &Tensor<In>,
    b: &Tensor<In>,
    out: &mut TensorValue,
) -> Result<()> {
    match out {
        TensorValue::U8(out) => add_accumulate_unsigned::<In, u8>(a, b, out),
        TensorValue::U16(out) => add_accumulate_unsigned::<In, u16>(a, b, out),
        TensorValue::U32(out) => add_accumulate_unsigned::<In, u32>(a, b, out),
        TensorValue::U64(out) => add_accumulate_unsigned::<In, u64>(a, b, out),
        _ => Err(anyhow!("dtype mismatch")),
    }
}

pub fn add_normal_dispatch(
    _attrs: &crate::graph::OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    let out = expect_output(output)?;
    add_normal_match!(&inputs[0], &inputs[1], out)
}

pub fn add_inplace_dispatch(
    _attrs: &crate::graph::OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    let out = expect_output(output)?;
    add_inplace_match!(out, &inputs[1])
}

pub fn add_accumulate_dispatch(
    _attrs: &crate::graph::OpAttrs,
    inputs: &[TensorValue],
    output: Option<&mut TensorValue>,
) -> Result<()> {
    let out = expect_output(output)?;
    add_accumulate_match!(&inputs[0], &inputs[1], out)
}
