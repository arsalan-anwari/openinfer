use anyhow::Result;

use crate::tensor::{broadcast_to_vec, Tensor, TensorOptions, TensorValue};

pub fn broadcast_value_to_shape(value: &TensorValue, out_shape: &[usize]) -> Result<TensorValue> {
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
    }
}
