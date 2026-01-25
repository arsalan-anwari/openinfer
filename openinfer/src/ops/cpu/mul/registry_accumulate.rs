use anyhow::anyhow;

use crate::graph::OpAttrs;
use crate::ops::KernelFn;
use crate::tensor::{DType, TensorElement, TensorValue};

use super::{
    mul_i16_i32,
    mul_i16_i32_broadcast,
    mul_i16_i64,
    mul_i16_i64_broadcast,
    mul_i32_i64,
    mul_i32_i64_broadcast,
    mul_i4_i16_packed,
    mul_i4_i16_packed_broadcast,
    mul_i4_i32_packed,
    mul_i4_i32_packed_broadcast,
    mul_i4_i64_packed,
    mul_i4_i64_packed_broadcast,
    mul_i4_i8_packed,
    mul_i4_i8_packed_broadcast,
    mul_i8_i16,
    mul_i8_i16_broadcast,
    mul_i8_i32,
    mul_i8_i32_broadcast,
    mul_i8_i64,
    mul_i8_i64_broadcast,
    mul_i2_i16_packed,
    mul_i2_i16_packed_broadcast,
    mul_i2_i32_packed,
    mul_i2_i32_packed_broadcast,
    mul_i2_i64_packed,
    mul_i2_i64_packed_broadcast,
    mul_i2_i8_packed,
    mul_i2_i8_packed_broadcast,
    mul_i1_i16_packed,
    mul_i1_i16_packed_broadcast,
    mul_i1_i32_packed,
    mul_i1_i32_packed_broadcast,
    mul_i1_i64_packed,
    mul_i1_i64_packed_broadcast,
    mul_i1_i8_packed,
    mul_i1_i8_packed_broadcast,
    mul_u16_u32,
    mul_u16_u32_broadcast,
    mul_u16_u64,
    mul_u16_u64_broadcast,
    mul_u32_u64,
    mul_u32_u64_broadcast,
    mul_u4_u16_packed,
    mul_u4_u16_packed_broadcast,
    mul_u4_u32_packed,
    mul_u4_u32_packed_broadcast,
    mul_u4_u64_packed,
    mul_u4_u64_packed_broadcast,
    mul_u4_u8_packed,
    mul_u4_u8_packed_broadcast,
    mul_u8_u16,
    mul_u8_u16_broadcast,
    mul_u8_u32,
    mul_u8_u32_broadcast,
    mul_u8_u64,
    mul_u8_u64_broadcast,
    mul_u2_u16_packed,
    mul_u2_u16_packed_broadcast,
    mul_u2_u32_packed,
    mul_u2_u32_packed_broadcast,
    mul_u2_u64_packed,
    mul_u2_u64_packed_broadcast,
    mul_u2_u8_packed,
    mul_u2_u8_packed_broadcast,
    mul_u1_u16_packed,
    mul_u1_u16_packed_broadcast,
    mul_u1_u32_packed,
    mul_u1_u32_packed_broadcast,
    mul_u1_u64_packed,
    mul_u1_u64_packed_broadcast,
    mul_u1_u8_packed,
    mul_u1_u8_packed_broadcast,
};

pub fn lookup_kernel_cpu_mul_accumulate(
    output_dtype: DType,
    input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<KernelFn> {
    match (output_dtype, input_dtypes, attrs) {
        (DType::I16, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(Accumulate, "mul", i8, I16, mul_i8_i16, mul_i8_i16_broadcast)
        }
        (DType::I32, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(Accumulate, "mul", i8, I32, mul_i8_i32, mul_i8_i32_broadcast)
        }
        (DType::I64, [DType::I8, DType::I8], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(Accumulate, "mul", i8, I64, mul_i8_i64, mul_i8_i64_broadcast)
        }
        (DType::I32, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(Accumulate, "mul", i16, I32, mul_i16_i32, mul_i16_i32_broadcast)
        }
        (DType::I64, [DType::I16, DType::I16], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(Accumulate, "mul", i16, I64, mul_i16_i64, mul_i16_i64_broadcast)
        }
        (DType::I64, [DType::I32, DType::I32], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(Accumulate, "mul", i32, I64, mul_i32_i64, mul_i32_i64_broadcast)
        }
        (DType::U16, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            crate::add_kernel!(Accumulate, "mul", u8, U16, mul_u8_u16, mul_u8_u16_broadcast)
        }
        (DType::U32, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            crate::add_kernel!(Accumulate, "mul", u8, U32, mul_u8_u32, mul_u8_u32_broadcast)
        }
        (DType::U64, [DType::U8, DType::U8], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(Accumulate, "mul", u8, U64, mul_u8_u64, mul_u8_u64_broadcast)
        }
        (DType::U32, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            crate::add_kernel!(Accumulate, "mul", u16, U32, mul_u16_u32, mul_u16_u32_broadcast)
        }
        (DType::U64, [DType::U16, DType::U16], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(Accumulate, "mul", u16, U64, mul_u16_u64, mul_u16_u64_broadcast)
        }
        (DType::U64, [DType::U32, DType::U32], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(Accumulate, "mul", u32, U64, mul_u32_u64, mul_u32_u64_broadcast)
        }
        (DType::I8, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::I4, I8, mul_i4_i8_packed, mul_i4_i8_packed_broadcast)
        }
        (DType::I16, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::I4, I16, mul_i4_i16_packed, mul_i4_i16_packed_broadcast)
        }
        (DType::I32, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::I4, I32, mul_i4_i32_packed, mul_i4_i32_packed_broadcast)
        }
        (DType::I64, [DType::I4, DType::I4], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::I4, I64, mul_i4_i64_packed, mul_i4_i64_packed_broadcast)
        }
        (DType::I8, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::I2, I8, mul_i2_i8_packed, mul_i2_i8_packed_broadcast)
        }
        (DType::I16, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::I2, I16, mul_i2_i16_packed, mul_i2_i16_packed_broadcast)
        }
        (DType::I32, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::I2, I32, mul_i2_i32_packed, mul_i2_i32_packed_broadcast)
        }
        (DType::I64, [DType::I2, DType::I2], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::I2, I64, mul_i2_i64_packed, mul_i2_i64_packed_broadcast)
        }
        (DType::I8, [DType::I1, DType::I1], &OpAttrs::Accumulate { dtype: DType::I8 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::I1, I8, mul_i1_i8_packed, mul_i1_i8_packed_broadcast)
        }
        (DType::I16, [DType::I1, DType::I1], &OpAttrs::Accumulate { dtype: DType::I16 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::I1, I16, mul_i1_i16_packed, mul_i1_i16_packed_broadcast)
        }
        (DType::I32, [DType::I1, DType::I1], &OpAttrs::Accumulate { dtype: DType::I32 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::I1, I32, mul_i1_i32_packed, mul_i1_i32_packed_broadcast)
        }
        (DType::I64, [DType::I1, DType::I1], &OpAttrs::Accumulate { dtype: DType::I64 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::I1, I64, mul_i1_i64_packed, mul_i1_i64_packed_broadcast)
        }
        (DType::U8, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U8 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::U4, U8, mul_u4_u8_packed, mul_u4_u8_packed_broadcast)
        }
        (DType::U16, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::U4, U16, mul_u4_u16_packed, mul_u4_u16_packed_broadcast)
        }
        (DType::U32, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::U4, U32, mul_u4_u32_packed, mul_u4_u32_packed_broadcast)
        }
        (DType::U64, [DType::U4, DType::U4], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::U4, U64, mul_u4_u64_packed, mul_u4_u64_packed_broadcast)
        }
        (DType::U8, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U8 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::U2, U8, mul_u2_u8_packed, mul_u2_u8_packed_broadcast)
        }
        (DType::U16, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::U2, U16, mul_u2_u16_packed, mul_u2_u16_packed_broadcast)
        }
        (DType::U32, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::U2, U32, mul_u2_u32_packed, mul_u2_u32_packed_broadcast)
        }
        (DType::U64, [DType::U2, DType::U2], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::U2, U64, mul_u2_u64_packed, mul_u2_u64_packed_broadcast)
        }
        (DType::U8, [DType::U1, DType::U1], &OpAttrs::Accumulate { dtype: DType::U8 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::U1, U8, mul_u1_u8_packed, mul_u1_u8_packed_broadcast)
        }
        (DType::U16, [DType::U1, DType::U1], &OpAttrs::Accumulate { dtype: DType::U16 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::U1, U16, mul_u1_u16_packed, mul_u1_u16_packed_broadcast)
        }
        (DType::U32, [DType::U1, DType::U1], &OpAttrs::Accumulate { dtype: DType::U32 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::U1, U32, mul_u1_u32_packed, mul_u1_u32_packed_broadcast)
        }
        (DType::U64, [DType::U1, DType::U1], &OpAttrs::Accumulate { dtype: DType::U64 }) => {
            crate::add_kernel!(Accumulate, "mul", crate::tensor::U1, U64, mul_u1_u64_packed, mul_u1_u64_packed_broadcast)
        }
        _ => None,
    }
}
