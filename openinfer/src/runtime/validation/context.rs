use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};

use crate::graph::{Graph, VarDecl};
use crate::runtime::ModelLoader;
use crate::tensor::{numel, DType};

#[derive(Debug)]
pub struct ValidationContext<'a> {
    pub model: &'a ModelLoader,
    pub graph: &'a Graph,
    pub var_shapes: HashMap<String, Vec<usize>>,
    pub var_dtypes: HashMap<String, DType>,
    pub temps: HashSet<String>,
}

impl<'a> ValidationContext<'a> {
    pub fn new(model: &'a ModelLoader, graph: &'a Graph) -> Self {
        Self {
            model,
            graph,
            var_shapes: HashMap::new(),
            var_dtypes: HashMap::new(),
            temps: HashSet::new(),
        }
    }

    pub fn register_decl(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Vec<usize>,
    ) -> Result<()> {
        if self.var_dtypes.contains_key(name) {
            return Err(anyhow!("duplicate variable declaration {}", name));
        }
        self.var_dtypes.insert(name.to_string(), dtype);
        self.var_shapes.insert(name.to_string(), shape);
        Ok(())
    }

    pub fn register_temp(&mut self, name: &str, dtype: DType, dims: &[String]) -> Result<()> {
        if self.graph.vars.contains_key(name) {
            return Err(anyhow!("assign shadows declared variable {}", name));
        }
        if self.temps.contains(name) {
            return Err(anyhow!("duplicate temporary {}", name));
        }
        let shape = self.model.resolve_shape(dims)?;
        self.var_dtypes.insert(name.to_string(), dtype);
        self.var_shapes.insert(name.to_string(), shape);
        self.temps.insert(name.to_string());
        Ok(())
    }

    pub fn decl_for(&self, name: &str) -> Option<&VarDecl> {
        let base = base_name(name);
        self.graph.vars.get(base)
    }

    pub fn has_var(&self, name: &str) -> bool {
        let base = base_name(name);
        self.var_dtypes.contains_key(base)
    }

    pub fn var_dtype(&self, name: &str) -> Result<DType> {
        let base = base_name(name);
        self.var_dtypes
            .get(base)
            .copied()
            .ok_or_else(|| anyhow!("unknown variable: {}", name))
    }

    pub fn var_shape(&self, name: &str) -> Result<Vec<usize>> {
        let base = base_name(name);
        self.var_shapes
            .get(base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown variable: {}", name))
    }

    pub fn is_scalar_var(&self, name: &str) -> Result<bool> {
        let shape = self.var_shape(name)?;
        Ok(numel(&shape) == 1)
    }
}

pub fn base_name(name: &str) -> &str {
    match name.split_once('[') {
        Some((base, _)) => base,
        None => name,
    }
}
