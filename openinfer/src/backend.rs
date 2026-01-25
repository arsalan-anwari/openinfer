use crate::tensor::{DType, TensorValue};

pub mod cpu;

#[derive(Debug, Clone)]
pub enum TensorStorage {
    Host(TensorValue),
}

// TensorStorage is moved across threads but not shared concurrently.
unsafe impl Send for TensorStorage {}

impl TensorStorage {
    #[allow(dead_code)]
    pub fn dtype(&self) -> DType {
        match self {
            TensorStorage::Host(value) => value.dtype(),
        }
    }
}
