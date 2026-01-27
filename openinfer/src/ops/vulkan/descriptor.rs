use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};

use crate::ops::cpu::broadcast::broadcast_strides;
use crate::tensor::{DType, TensorValue};

include!(concat!(env!("OUT_DIR"), "/vulkan_config.rs"));

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct TensorDesc {
    pub rank: u32,
    pub dtype: u32,
    pub elem_bits: u32,
    pub byte_offset: u32,
    pub shape: [u32; MAX_DIMS],
    pub strides: [u32; MAX_DIMS],
}

pub fn dtype_code(dtype: DType) -> u32 {
    match dtype {
        DType::I8 => 1,
        DType::I16 => 2,
        DType::I32 => 3,
        DType::I64 => 4,
        DType::U8 => 5,
        DType::U16 => 6,
        DType::U32 => 7,
        DType::U64 => 8,
        DType::F16 => 9,
        DType::BF16 => 10,
        DType::F32 => 11,
        DType::F64 => 12,
        DType::F8E5M2 => 13,
        DType::Bool => 14,
        DType::Bitset => 15,
        DType::I4 => 16,
        DType::I2 => 17,
        DType::I1 => 18,
        DType::U4 => 19,
        DType::U2 => 20,
        DType::U1 => 21,
        DType::T1 => 22,
        DType::T2 => 23,
    }
}

pub fn build_tensor_desc(
    value: &TensorValue,
    out_rank: usize,
    byte_offset: u32,
) -> Result<TensorDesc> {
    let dtype = value.dtype();
    let shape = value.shape();
    if out_rank > MAX_DIMS {
        return Err(anyhow!(
            "vulkan tensors only support up to {} dims (got {})",
            MAX_DIMS,
            out_rank
        ));
    }
    let strides = broadcast_strides(shape, value.strides(), out_rank);
    let mut desc = TensorDesc::default();
    desc.rank = out_rank as u32;
    desc.dtype = dtype_code(dtype);
    desc.elem_bits = dtype.bit_width() as u32;
    desc.byte_offset = byte_offset;
    for i in 0..out_rank {
        desc.shape[i] = shape.get(i).copied().unwrap_or(1) as u32;
        desc.strides[i] = strides[i] as u32;
    }
    Ok(desc)
}
