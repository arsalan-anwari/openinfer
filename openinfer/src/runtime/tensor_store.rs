use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use memmap2::Mmap;

use crate::tensor::DType;

/// A slice into a memory-mapped model file.
#[derive(Debug, Clone)]
pub struct MappedSlice {
    mmap: Arc<Mmap>,
    range: Range<usize>,
}

impl MappedSlice {
    /// Create a mapped slice for a byte range.
    pub fn new(mmap: Arc<Mmap>, range: Range<usize>) -> Self {
        Self { mmap, range }
    }

    /// Return the raw bytes for this slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap[self.range.clone()]
    }
}

/// Metadata describing a tensor stored in a model file.
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub name: String,
    pub dtype: DType,
    pub dims: Vec<String>,
    pub shape: Vec<usize>,
    pub data: Option<MappedSlice>,
}

impl TensorRef {
    /// Build a human-readable description string.
    pub fn describe(&self) -> String {
        format!("{}:{:?}{:?}", self.name, self.dtype, self.shape)
    }
}

/// Collection of named tensor references.
#[derive(Debug, Clone)]
pub struct TensorStore {
    tensors: HashMap<String, TensorRef>,
}

impl TensorStore {
    /// Create a store from a set of tensors.
    pub fn new(tensors: HashMap<String, TensorRef>) -> Self {
        Self { tensors }
    }

    /// Fetch a tensor reference by name.
    pub fn get(&self, name: &str) -> Result<&TensorRef> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("unknown tensor: {}", name))
    }

    /// Insert or replace a tensor reference.
    pub fn insert(&mut self, tensor: TensorRef) {
        self.tensors.insert(tensor.name.clone(), tensor);
    }

    /// Check whether a tensor name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}
