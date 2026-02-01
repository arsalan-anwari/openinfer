use anyhow::{anyhow, Result};

use crate::tensor::Tensor;

fn popcount_u64(value: u64) -> u8 {
    value.count_ones() as u8
}

fn popcount_u32(value: u32) -> u8 {
    value.count_ones() as u8
}

fn popcount_u16(value: u16) -> u8 {
    value.count_ones() as u8
}

fn popcount_u8(value: u8) -> u8 {
    value.count_ones() as u8
}

pub fn popcount_i8_normal(a: &Tensor<i8>, out: &mut Tensor<u8>) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        *out_slot = popcount_u8(*value as u8);
    }
    Ok(())
}

pub fn popcount_i16_normal(a: &Tensor<i16>, out: &mut Tensor<u8>) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        *out_slot = popcount_u16(*value as u16);
    }
    Ok(())
}

pub fn popcount_i32_normal(a: &Tensor<i32>, out: &mut Tensor<u8>) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        *out_slot = popcount_u32(*value as u32);
    }
    Ok(())
}

pub fn popcount_i64_normal(a: &Tensor<i64>, out: &mut Tensor<u8>) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        *out_slot = popcount_u64(*value as u64);
    }
    Ok(())
}

pub fn popcount_u8_normal(a: &Tensor<u8>, out: &mut Tensor<u8>) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        *out_slot = popcount_u8(*value);
    }
    Ok(())
}

pub fn popcount_u16_normal(a: &Tensor<u16>, out: &mut Tensor<u8>) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        *out_slot = popcount_u16(*value);
    }
    Ok(())
}

pub fn popcount_u32_normal(a: &Tensor<u32>, out: &mut Tensor<u8>) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        *out_slot = popcount_u32(*value);
    }
    Ok(())
}

pub fn popcount_u64_normal(a: &Tensor<u64>, out: &mut Tensor<u8>) -> Result<()> {
    if a.shape() != out.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        *out_slot = popcount_u64(*value);
    }
    Ok(())
}
