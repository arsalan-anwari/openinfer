use anyhow::{anyhow, Result};

use crate::runtime::state::RuntimeState;
use crate::runtime::value_eval::{tensor_to_bool, tensor_to_i64};
use crate::tensor::{Tensor, TensorElement, TensorValue};

pub trait Fetchable: Sized {
    fn fetch(state: &mut RuntimeState, name: &str) -> Result<Self>;
}

impl Fetchable for TensorValue {
    fn fetch(state: &mut RuntimeState, name: &str) -> Result<Self> {
        state.get_tensor(name)
    }
}

impl<T: TensorElement> Fetchable for Tensor<T> {
    fn fetch(state: &mut RuntimeState, name: &str) -> Result<Self> {
        state.fetch_typed(name)
    }
}

macro_rules! impl_fetch_int {
    ($t:ty) => {
        impl Fetchable for $t {
            fn fetch(state: &mut RuntimeState, name: &str) -> Result<Self> {
                let tensor: TensorValue = state.get_tensor(name)?;
                Ok(tensor_to_i64(&tensor)? as $t)
            }
        }
    };
}

macro_rules! impl_fetch_float {
    ($t:ty) => {
        impl Fetchable for $t {
            fn fetch(state: &mut RuntimeState, name: &str) -> Result<Self> {
                let tensor: TensorValue = state.get_tensor(name)?;
                Ok(tensor_to_f64(&tensor)? as $t)
            }
        }
    };
}

impl_fetch_float!(f32);
impl_fetch_float!(f64);
impl_fetch_int!(i8);
impl_fetch_int!(i16);
impl_fetch_int!(i32);
impl_fetch_int!(i64);
impl_fetch_int!(u8);
impl_fetch_int!(u16);
impl_fetch_int!(u32);
impl_fetch_int!(u64);

impl Fetchable for bool {
    fn fetch(state: &mut RuntimeState, name: &str) -> Result<Self> {
        let tensor: TensorValue = state.get_tensor(name)?;
        tensor_to_bool(&tensor)
    }
}

fn tensor_to_f64(value: &TensorValue) -> Result<f64> {
    if value.len() != 1 {
        return Err(anyhow!("expected scalar value"));
    }
    match value {
        TensorValue::I8(t) => Ok(t.data[0] as f64),
        TensorValue::I16(t) => Ok(t.data[0] as f64),
        TensorValue::I32(t) => Ok(t.data[0] as f64),
        TensorValue::I64(t) => Ok(t.data[0] as f64),
        TensorValue::U8(t) => Ok(t.data[0] as f64),
        TensorValue::U16(t) => Ok(t.data[0] as f64),
        TensorValue::U32(t) => Ok(t.data[0] as f64),
        TensorValue::U64(t) => Ok(t.data[0] as f64),
        TensorValue::Bool(t) => Ok(if t.data[0] { 1.0 } else { 0.0 }),
        TensorValue::F16(t) => Ok(t.data[0].to_f32() as f64),
        TensorValue::BF16(t) => Ok(t.data[0].to_f32() as f64),
        TensorValue::F8(t) => Ok(t.data[0].to_f32() as f64),
        TensorValue::F32(t) => Ok(t.data[0] as f64),
        TensorValue::F64(t) => Ok(t.data[0]),
        TensorValue::Bitset(t) => Ok(t.data[0].bits as f64),
        TensorValue::I4(_)
        | TensorValue::I2(_)
        | TensorValue::I1(_)
        | TensorValue::U4(_)
        | TensorValue::U2(_)
        | TensorValue::U1(_)
        | TensorValue::T2(_)
        | TensorValue::T1(_) => Err(anyhow!("packed scalars are not supported")),
    }
}
