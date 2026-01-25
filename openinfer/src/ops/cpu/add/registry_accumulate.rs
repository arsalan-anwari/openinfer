use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, TensorElement, TensorValue};

use super::{
    add_i16_i32,
    add_i16_i32_broadcast,
    add_i16_i64,
    add_i16_i64_broadcast,
    add_i32_i64,
    add_i32_i64_broadcast,
    add_i4_i16_packed,
    add_i4_i16_packed_broadcast,
    add_i4_i32_packed,
    add_i4_i32_packed_broadcast,
    add_i4_i64_packed,
    add_i4_i64_packed_broadcast,
    add_i4_i8_packed,
    add_i4_i8_packed_broadcast,
    add_i8_i16,
    add_i8_i16_broadcast,
    add_i8_i32,
    add_i8_i32_broadcast,
    add_i8_i64,
    add_i8_i64_broadcast,
    add_i2_i16_packed,
    add_i2_i16_packed_broadcast,
    add_i2_i32_packed,
    add_i2_i32_packed_broadcast,
    add_i2_i64_packed,
    add_i2_i64_packed_broadcast,
    add_i2_i8_packed,
    add_i2_i8_packed_broadcast,
    add_i1_i16_packed,
    add_i1_i16_packed_broadcast,
    add_i1_i32_packed,
    add_i1_i32_packed_broadcast,
    add_i1_i64_packed,
    add_i1_i64_packed_broadcast,
    add_i1_i8_packed,
    add_i1_i8_packed_broadcast,
    add_u16_u32,
    add_u16_u32_broadcast,
    add_u16_u64,
    add_u16_u64_broadcast,
    add_u32_u64,
    add_u32_u64_broadcast,
    add_u4_u16_packed,
    add_u4_u16_packed_broadcast,
    add_u4_u32_packed,
    add_u4_u32_packed_broadcast,
    add_u4_u64_packed,
    add_u4_u64_packed_broadcast,
    add_u4_u8_packed,
    add_u4_u8_packed_broadcast,
    add_u8_u16,
    add_u8_u16_broadcast,
    add_u8_u32,
    add_u8_u32_broadcast,
    add_u8_u64,
    add_u8_u64_broadcast,
    add_u2_u16_packed,
    add_u2_u16_packed_broadcast,
    add_u2_u32_packed,
    add_u2_u32_packed_broadcast,
    add_u2_u64_packed,
    add_u2_u64_packed_broadcast,
    add_u2_u8_packed,
    add_u2_u8_packed_broadcast,
    add_u1_u16_packed,
    add_u1_u16_packed_broadcast,
    add_u1_u32_packed,
    add_u1_u32_packed_broadcast,
    add_u1_u64_packed,
    add_u1_u64_packed_broadcast,
    add_u1_u8_packed,
    add_u1_u8_packed_broadcast,
};

pub fn lookup_kernel_cpu_add_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I16, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(Accumulate, "add", i8, I16, add_i8_i16, add_i8_i16_broadcast)
        }
        (DType::I32, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(Accumulate, "add", i8, I32, add_i8_i32, add_i8_i32_broadcast)
        }
        (DType::I64, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(Accumulate, "add", i8, I64, add_i8_i64, add_i8_i64_broadcast)
        }
        (DType::I32, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(Accumulate, "add", i16, I32, add_i16_i32, add_i16_i32_broadcast)
        }
        (DType::I64, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(Accumulate, "add", i16, I64, add_i16_i64, add_i16_i64_broadcast)
        }
        (DType::I64, [DType::I32, DType::I32], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(Accumulate, "add", i32, I64, add_i32_i64, add_i32_i64_broadcast)
        }
        (DType::U16, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            crate::add_kernel!(Accumulate, "add", u8, U16, add_u8_u16, add_u8_u16_broadcast)
        }
        (DType::U32, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            crate::add_kernel!(Accumulate, "add", u8, U32, add_u8_u32, add_u8_u32_broadcast)
        }
        (DType::U64, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(Accumulate, "add", u8, U64, add_u8_u64, add_u8_u64_broadcast)
        }
        (DType::U32, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            crate::add_kernel!(Accumulate, "add", u16, U32, add_u16_u32, add_u16_u32_broadcast)
        }
        (DType::U64, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(Accumulate, "add", u16, U64, add_u16_u64, add_u16_u64_broadcast)
        }
        (DType::U64, [DType::U32, DType::U32], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(Accumulate, "add", u32, U64, add_u32_u64, add_u32_u64_broadcast)
        }
        (DType::I8, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::I4, I8, add_i4_i8_packed, add_i4_i8_packed_broadcast)
        }
        (DType::I16, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::I4, I16, add_i4_i16_packed, add_i4_i16_packed_broadcast)
        }
        (DType::I32, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::I4, I32, add_i4_i32_packed, add_i4_i32_packed_broadcast)
        }
        (DType::I64, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::I4, I64, add_i4_i64_packed, add_i4_i64_packed_broadcast)
        }
        (DType::I8, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::I2, I8, add_i2_i8_packed, add_i2_i8_packed_broadcast)
        }
        (DType::I16, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::I2, I16, add_i2_i16_packed, add_i2_i16_packed_broadcast)
        }
        (DType::I32, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::I2, I32, add_i2_i32_packed, add_i2_i32_packed_broadcast)
        }
        (DType::I64, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::I2, I64, add_i2_i64_packed, add_i2_i64_packed_broadcast)
        }
        (DType::I8, [DType::I1, DType::I1], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::I1, I8, add_i1_i8_packed, add_i1_i8_packed_broadcast)
        }
        (DType::I16, [DType::I1, DType::I1], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::I1, I16, add_i1_i16_packed, add_i1_i16_packed_broadcast)
        }
        (DType::I32, [DType::I1, DType::I1], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::I1, I32, add_i1_i32_packed, add_i1_i32_packed_broadcast)
        }
        (DType::I64, [DType::I1, DType::I1], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::I1, I64, add_i1_i64_packed, add_i1_i64_packed_broadcast)
        }
        (DType::U8, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U8 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::U4, U8, add_u4_u8_packed, add_u4_u8_packed_broadcast)
        }
        (DType::U16, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::U4, U16, add_u4_u16_packed, add_u4_u16_packed_broadcast)
        }
        (DType::U32, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::U4, U32, add_u4_u32_packed, add_u4_u32_packed_broadcast)
        }
        (DType::U64, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::U4, U64, add_u4_u64_packed, add_u4_u64_packed_broadcast)
        }
        (DType::U8, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U8 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::U2, U8, add_u2_u8_packed, add_u2_u8_packed_broadcast)
        }
        (DType::U16, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::U2, U16, add_u2_u16_packed, add_u2_u16_packed_broadcast)
        }
        (DType::U32, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::U2, U32, add_u2_u32_packed, add_u2_u32_packed_broadcast)
        }
        (DType::U64, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::U2, U64, add_u2_u64_packed, add_u2_u64_packed_broadcast)
        }
        (DType::U8, [DType::U1, DType::U1], &OpAttrs::Accumulate { dtype: DType::U8 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::U1, U8, add_u1_u8_packed, add_u1_u8_packed_broadcast)
        }
        (DType::U16, [DType::U1, DType::U1], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::U1, U16, add_u1_u16_packed, add_u1_u16_packed_broadcast)
        }
        (DType::U32, [DType::U1, DType::U1], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::U1, U32, add_u1_u32_packed, add_u1_u32_packed_broadcast)
        }
        (DType::U64, [DType::U1, DType::U1], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(Accumulate, "add", crate::tensor::U1, U64, add_u1_u64_packed, add_u1_u64_packed_broadcast)
        }
        _ => None,
    }
}
