use anyhow::{anyhow, Result};

use crate::tensor::{Bitset, BF16, F16, F8E5M2, I1, I2, I4, U1, U2, U4, TensorValue};

use super::accumulate::{add_accumulate_signed, add_accumulate_unsigned};
use super::kernel::{add_inplace, add_normal};
use super::packed::{
    add_packed_signed, add_packed_signed_inplace, add_packed_unsigned, add_packed_unsigned_inplace,
};

macro_rules! expect_tensor_ref {
    ($value:expr, $variant:ident) => {
        match $value {
            TensorValue::$variant(tensor) => Ok(tensor),
            _ => Err(anyhow!("dtype mismatch")),
        }
    };
}

macro_rules! expect_tensor_mut {
    ($value:expr, $variant:ident) => {
        match $value {
            TensorValue::$variant(tensor) => Ok(tensor),
            _ => Err(anyhow!("dtype mismatch")),
        }
    };
}

macro_rules! define_add_wrappers {
    ($normal_fn:ident, $inplace_fn:ident, $variant:ident, $ty:ty) => {
        pub fn $normal_fn(
            _attrs: &crate::graph::OpAttrs,
            inputs: &[TensorValue],
            output: Option<&mut TensorValue>,
        ) -> Result<()> {
            let a = expect_tensor_ref!(&inputs[0], $variant)?;
            let b = expect_tensor_ref!(&inputs[1], $variant)?;
            let out = output.ok_or_else(|| anyhow!("missing output tensor"))?;
            let out_tensor = expect_tensor_mut!(out, $variant)?;
            add_normal::<$ty>(a, b, out_tensor)
        }

        pub fn $inplace_fn(
            _attrs: &crate::graph::OpAttrs,
            inputs: &[TensorValue],
            output: Option<&mut TensorValue>,
        ) -> Result<()> {
            let out = output.ok_or_else(|| anyhow!("missing output tensor"))?;
            let a = expect_tensor_mut!(out, $variant)?;
            let b = expect_tensor_ref!(&inputs[1], $variant)?;
            add_inplace::<$ty>(a, b)
        }
    };
}

macro_rules! define_add_wrappers_packed_signed {
    ($normal_fn:ident, $inplace_fn:ident, $variant:ident, $ty:ty, $width:expr) => {
        pub fn $normal_fn(
            _attrs: &crate::graph::OpAttrs,
            inputs: &[TensorValue],
            output: Option<&mut TensorValue>,
        ) -> Result<()> {
            let a = expect_tensor_ref!(&inputs[0], $variant)?;
            let b = expect_tensor_ref!(&inputs[1], $variant)?;
            let out = output.ok_or_else(|| anyhow!("missing output tensor"))?;
            let out_tensor = expect_tensor_mut!(out, $variant)?;
            add_packed_signed::<$ty>(a, b, out_tensor, $width)
        }

        pub fn $inplace_fn(
            _attrs: &crate::graph::OpAttrs,
            inputs: &[TensorValue],
            output: Option<&mut TensorValue>,
        ) -> Result<()> {
            let out = output.ok_or_else(|| anyhow!("missing output tensor"))?;
            let a = expect_tensor_mut!(out, $variant)?;
            let b = expect_tensor_ref!(&inputs[1], $variant)?;
            add_packed_signed_inplace::<$ty>(a, b, $width)
        }
    };
}

macro_rules! define_add_wrappers_packed_unsigned {
    ($normal_fn:ident, $inplace_fn:ident, $variant:ident, $ty:ty, $width:expr) => {
        pub fn $normal_fn(
            _attrs: &crate::graph::OpAttrs,
            inputs: &[TensorValue],
            output: Option<&mut TensorValue>,
        ) -> Result<()> {
            let a = expect_tensor_ref!(&inputs[0], $variant)?;
            let b = expect_tensor_ref!(&inputs[1], $variant)?;
            let out = output.ok_or_else(|| anyhow!("missing output tensor"))?;
            let out_tensor = expect_tensor_mut!(out, $variant)?;
            add_packed_unsigned::<$ty>(a, b, out_tensor, $width)
        }

        pub fn $inplace_fn(
            _attrs: &crate::graph::OpAttrs,
            inputs: &[TensorValue],
            output: Option<&mut TensorValue>,
        ) -> Result<()> {
            let out = output.ok_or_else(|| anyhow!("missing output tensor"))?;
            let a = expect_tensor_mut!(out, $variant)?;
            let b = expect_tensor_ref!(&inputs[1], $variant)?;
            add_packed_unsigned_inplace::<$ty>(a, b, $width)
        }
    };
}

macro_rules! define_add_accumulate_signed {
    ($fn_name:ident, $in_variant:ident, $in_ty:ty, $out_variant:ident, $out_ty:ty) => {
        pub fn $fn_name(
            _attrs: &crate::graph::OpAttrs,
            inputs: &[TensorValue],
            output: Option<&mut TensorValue>,
        ) -> Result<()> {
            let a = expect_tensor_ref!(&inputs[0], $in_variant)?;
            let b = expect_tensor_ref!(&inputs[1], $in_variant)?;
            let out = output.ok_or_else(|| anyhow!("missing output tensor"))?;
            let out_tensor = expect_tensor_mut!(out, $out_variant)?;
            add_accumulate_signed::<$in_ty, $out_ty>(a, b, out_tensor)
        }
    };
}

macro_rules! define_add_accumulate_unsigned {
    ($fn_name:ident, $in_variant:ident, $in_ty:ty, $out_variant:ident, $out_ty:ty) => {
        pub fn $fn_name(
            _attrs: &crate::graph::OpAttrs,
            inputs: &[TensorValue],
            output: Option<&mut TensorValue>,
        ) -> Result<()> {
            let a = expect_tensor_ref!(&inputs[0], $in_variant)?;
            let b = expect_tensor_ref!(&inputs[1], $in_variant)?;
            let out = output.ok_or_else(|| anyhow!("missing output tensor"))?;
            let out_tensor = expect_tensor_mut!(out, $out_variant)?;
            add_accumulate_unsigned::<$in_ty, $out_ty>(a, b, out_tensor)
        }
    };
}

define_add_wrappers!(add_normal_f8, add_inplace_f8, F8E5M2, F8E5M2);
define_add_wrappers!(add_normal_bf16, add_inplace_bf16, BF16, BF16);
define_add_wrappers!(add_normal_f16, add_inplace_f16, F16, F16);
define_add_wrappers!(add_normal_f32, add_inplace_f32, F32, f32);
define_add_wrappers!(add_normal_f64, add_inplace_f64, F64, f64);

define_add_wrappers_packed_signed!(add_normal_i1, add_inplace_i1, I1, I1, 1);
define_add_wrappers_packed_signed!(add_normal_i2, add_inplace_i2, I2, I2, 2);
define_add_wrappers_packed_signed!(add_normal_i4, add_inplace_i4, I4, I4, 4);
define_add_wrappers!(add_normal_i8, add_inplace_i8, I8, i8);
define_add_wrappers!(add_normal_i16, add_inplace_i16, I16, i16);
define_add_wrappers!(add_normal_i32, add_inplace_i32, I32, i32);
define_add_wrappers!(add_normal_i64, add_inplace_i64, I64, i64);

define_add_wrappers_packed_unsigned!(add_normal_u1, add_inplace_u1, U1, U1, 1);
define_add_wrappers_packed_unsigned!(add_normal_u2, add_inplace_u2, U2, U2, 2);
define_add_wrappers_packed_unsigned!(add_normal_u4, add_inplace_u4, U4, U4, 4);
define_add_wrappers!(add_normal_u8, add_inplace_u8, U8, u8);
define_add_wrappers!(add_normal_u16, add_inplace_u16, U16, u16);
define_add_wrappers!(add_normal_u32, add_inplace_u32, U32, u32);
define_add_wrappers!(add_normal_u64, add_inplace_u64, U64, u64);

define_add_wrappers!(add_normal_bool, add_inplace_bool, Bool, bool);
define_add_wrappers!(add_normal_bitset, add_inplace_bitset, Bitset, Bitset);

define_add_accumulate_signed!(add_accumulate_i8_i8, I8, i8, I8, i8);
define_add_accumulate_signed!(add_accumulate_i8_i16, I8, i8, I16, i16);
define_add_accumulate_signed!(add_accumulate_i8_i32, I8, i8, I32, i32);
define_add_accumulate_signed!(add_accumulate_i8_i64, I8, i8, I64, i64);
define_add_accumulate_signed!(add_accumulate_i16_i8, I16, i16, I8, i8);
define_add_accumulate_signed!(add_accumulate_i16_i16, I16, i16, I16, i16);
define_add_accumulate_signed!(add_accumulate_i16_i32, I16, i16, I32, i32);
define_add_accumulate_signed!(add_accumulate_i16_i64, I16, i16, I64, i64);
define_add_accumulate_signed!(add_accumulate_i32_i8, I32, i32, I8, i8);
define_add_accumulate_signed!(add_accumulate_i32_i16, I32, i32, I16, i16);
define_add_accumulate_signed!(add_accumulate_i32_i32, I32, i32, I32, i32);
define_add_accumulate_signed!(add_accumulate_i32_i64, I32, i32, I64, i64);
define_add_accumulate_signed!(add_accumulate_i64_i8, I64, i64, I8, i8);
define_add_accumulate_signed!(add_accumulate_i64_i16, I64, i64, I16, i16);
define_add_accumulate_signed!(add_accumulate_i64_i32, I64, i64, I32, i32);
define_add_accumulate_signed!(add_accumulate_i64_i64, I64, i64, I64, i64);

define_add_accumulate_unsigned!(add_accumulate_u8_u8, U8, u8, U8, u8);
define_add_accumulate_unsigned!(add_accumulate_u8_u16, U8, u8, U16, u16);
define_add_accumulate_unsigned!(add_accumulate_u8_u32, U8, u8, U32, u32);
define_add_accumulate_unsigned!(add_accumulate_u8_u64, U8, u8, U64, u64);
define_add_accumulate_unsigned!(add_accumulate_u16_u8, U16, u16, U8, u8);
define_add_accumulate_unsigned!(add_accumulate_u16_u16, U16, u16, U16, u16);
define_add_accumulate_unsigned!(add_accumulate_u16_u32, U16, u16, U32, u32);
define_add_accumulate_unsigned!(add_accumulate_u16_u64, U16, u16, U64, u64);
define_add_accumulate_unsigned!(add_accumulate_u32_u8, U32, u32, U8, u8);
define_add_accumulate_unsigned!(add_accumulate_u32_u16, U32, u32, U16, u16);
define_add_accumulate_unsigned!(add_accumulate_u32_u32, U32, u32, U32, u32);
define_add_accumulate_unsigned!(add_accumulate_u32_u64, U32, u32, U64, u64);
define_add_accumulate_unsigned!(add_accumulate_u64_u8, U64, u64, U8, u8);
define_add_accumulate_unsigned!(add_accumulate_u64_u16, U64, u64, U16, u16);
define_add_accumulate_unsigned!(add_accumulate_u64_u32, U64, u64, U32, u32);
define_add_accumulate_unsigned!(add_accumulate_u64_u64, U64, u64, U64, u64);
