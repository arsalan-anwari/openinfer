use anyhow::{anyhow, Result};

use crate::tensor::{broadcast_to_vec, Tensor, TensorOptions, TensorValue};

pub fn broadcast_value_to_shape(value: &TensorValue, out_shape: &[usize]) -> Result<TensorValue> {
    if value.dtype().is_packed() {
        if value.shape() == out_shape {
            return Ok(value.clone());
        }
        return Err(anyhow!(
            "broadcast not supported for packed dtype {:?}",
            value.dtype()
        ));
    }
    macro_rules! broadcast_value {
        ($tensor:expr, $variant:ident) => {{
            let data = broadcast_to_vec($tensor, out_shape)?;
            let tensor = Tensor::from_vec_with_opts(
                data,
                TensorOptions {
                    shape: Some(out_shape.to_vec()),
                    ..TensorOptions::default()
                },
            )?;
            Ok(TensorValue::$variant(tensor))
        }};
    }
    match value {
        TensorValue::I8(tensor) => broadcast_value!(tensor, I8),
        TensorValue::I16(tensor) => broadcast_value!(tensor, I16),
        TensorValue::F32(tensor) => broadcast_value!(tensor, F32),
        TensorValue::F64(tensor) => broadcast_value!(tensor, F64),
        TensorValue::U8(tensor) => broadcast_value!(tensor, U8),
        TensorValue::U16(tensor) => broadcast_value!(tensor, U16),
        TensorValue::I32(tensor) => broadcast_value!(tensor, I32),
        TensorValue::I64(tensor) => broadcast_value!(tensor, I64),
        TensorValue::U32(tensor) => broadcast_value!(tensor, U32),
        TensorValue::U64(tensor) => broadcast_value!(tensor, U64),
        TensorValue::Bool(tensor) => broadcast_value!(tensor, Bool),
        TensorValue::Bitset(tensor) => broadcast_value!(tensor, Bitset),
        TensorValue::F16(tensor) => broadcast_value!(tensor, F16),
        TensorValue::BF16(tensor) => broadcast_value!(tensor, BF16),
        TensorValue::F8E5M2(tensor) => broadcast_value!(tensor, F8E5M2),
        TensorValue::I4(tensor) => broadcast_value!(tensor, I4),
        TensorValue::I2(tensor) => broadcast_value!(tensor, I2),
        TensorValue::I1(tensor) => broadcast_value!(tensor, I1),
        TensorValue::U4(tensor) => broadcast_value!(tensor, U4),
        TensorValue::U2(tensor) => broadcast_value!(tensor, U2),
        TensorValue::U1(tensor) => broadcast_value!(tensor, U1),
        TensorValue::T2(tensor) => broadcast_value!(tensor, T2),
        TensorValue::T1(tensor) => broadcast_value!(tensor, T1),
    }
}
