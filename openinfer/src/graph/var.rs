use serde::{Deserialize, Serialize};

use crate::tensor::{DType, ScalarValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryKind {
    Dynamic,
    Volatile,
    Constant,
    Persistent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarDecl {
    pub name: String,
    #[serde(default)]
    pub ref_name: Option<String>,
    #[serde(default)]
    pub pattern: Option<String>,
    #[serde(default)]
    pub table_indices: Vec<String>,
    #[serde(default)]
    pub table: bool,
    #[serde(default)]
    pub auto_dim: Vec<String>,
    #[serde(default)]
    pub fixed: Vec<(String, usize)>,
    pub dtype: DType,
    pub dims: Vec<String>,
    pub kind: MemoryKind,
    pub init: Option<ScalarValue>,
}

impl VarDecl {
    pub fn model_name(&self) -> &str {
        self.ref_name.as_deref().unwrap_or(&self.name)
    }

    pub fn is_prefix_table(&self) -> bool {
        self.pattern.is_some()
    }

    pub fn is_cache_table(&self) -> bool {
        self.kind == MemoryKind::Persistent && self.table && !self.table_indices.is_empty()
    }

    pub fn has_auto_dim(&self) -> bool {
        self.kind == MemoryKind::Persistent && !self.auto_dim.is_empty()
    }

    pub fn cache_table_indices(&self) -> Vec<String> {
        if !self.is_cache_table() {
            return Vec::new();
        }
        self.table_indices
            .iter()
            .filter(|index| !self.auto_dim.contains(index))
            .cloned()
            .collect()
    }
}
