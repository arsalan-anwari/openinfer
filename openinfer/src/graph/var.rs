use serde::{Deserialize, Serialize};

use crate::tensor::{DType, ScalarValue};

/// Variable storage classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryKind {
    Dynamic,
    Volatile,
    Constant,
    Persistent,
}

/// Variable declaration within a graph.
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
    /// Return the model-facing name (alias or own name).
    pub fn model_name(&self) -> &str {
        self.ref_name.as_deref().unwrap_or(&self.name)
    }

    /// True if this variable is a prefix table.
    pub fn is_prefix_table(&self) -> bool {
        self.pattern.is_some()
    }

    /// True if this variable represents a cache table.
    pub fn is_cache_table(&self) -> bool {
        self.kind == MemoryKind::Persistent && self.table && !self.table_indices.is_empty()
    }

    /// True if this variable has auto-dimension entries.
    pub fn has_auto_dim(&self) -> bool {
        self.kind == MemoryKind::Persistent && !self.auto_dim.is_empty()
    }

    /// Cache table indices excluding auto-dimension indices.
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
