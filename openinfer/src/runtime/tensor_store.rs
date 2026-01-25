use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use memmap2::Mmap;

use crate::tensor::DType;

#[derive(Debug, Clone)]
pub struct MappedSlice {
    mmap: Arc<Mmap>,
    range: Range<usize>,
}

impl MappedSlice {
    pub fn new(mmap: Arc<Mmap>, range: Range<usize>) -> Self {
        Self { mmap, range }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap[self.range.clone()]
    }
}

#[derive(Debug, Clone)]
pub struct TensorRef {
    pub name: String,
    pub dtype: DType,
    pub dims: Vec<String>,
    pub shape: Vec<usize>,
    pub data: Option<MappedSlice>,
}

impl TensorRef {
    pub fn describe(&self) -> String {
        format!("{}:{:?}{:?}", self.name, self.dtype, self.shape)
    }
}

#[derive(Debug, Clone)]
pub struct TensorStore {
    tensors: HashMap<String, TensorRef>,
}

impl TensorStore {
    pub fn new(tensors: HashMap<String, TensorRef>) -> Self {
        Self { tensors }
    }

    pub fn get(&self, name: &str) -> Result<&TensorRef> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("unknown tensor: {}", name))
    }

    pub fn insert(&mut self, tensor: TensorRef) {
        self.tensors.insert(tensor.name.clone(), tensor);
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}
