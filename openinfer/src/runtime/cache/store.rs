use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::graph::{CacheAccess, CacheIndexExpr, CacheIndexValue, Graph, MemoryKind, VarDecl};
use crate::runtime::model_loader::ModelLoader;
use crate::tensor::TensorValue;

use super::auto_dim::{auto_dim_shape, fixed_size_for, AutoDimState};
use super::table::{
    build_cache_table, read_table_selection, recompute_cache_table_sizes, reset_cache_table_sizes,
    resolve_table_indices_from_resolved, resolve_table_prefix_indices_from_resolved, CacheTable,
};
use super::utils::{expand_tensor_value, increment_scalar, scalar_to_i64, slice_tensor_value};
use super::{cache_index_is_single, ResolvedCacheIndexExpr, TensorIndexSelection};

#[derive(Debug, Clone)]
pub struct CacheStore {
    persistent: HashMap<String, TensorValue>,
    tables: HashMap<String, CacheTable>,
    auto_dims: HashMap<String, AutoDimState>,
}

impl CacheStore {
    pub fn new(graph: &Graph, model: &ModelLoader) -> Result<Self> {
        let mut tables = HashMap::new();
        let mut auto_dims = HashMap::new();
        for (name, decl) in &graph.vars {
            if decl.is_cache_table() {
                let table = build_cache_table(decl)?;
                tables.insert(name.clone(), table);
            }
            if decl.has_auto_dim() {
                let base_shape = model.resolve_shape(&decl.dims)?;
                let max = decl
                    .auto_dim
                    .iter()
                    .map(|index| fixed_size_for(decl, index))
                    .collect::<Vec<_>>();
                auto_dims.insert(
                    name.clone(),
                    AutoDimState {
                        base_shape,
                        counts: vec![0; decl.auto_dim.len()],
                        max,
                    },
                );
            }
        }
        Ok(Self {
            persistent: HashMap::new(),
            tables,
            auto_dims,
        })
    }

    pub fn has_persistent(&self, name: &str) -> bool {
        self.persistent.contains_key(name)
    }

    pub fn get_persistent(&self, name: &str) -> Option<TensorValue> {
        self.persistent.get(name).cloned()
    }

    pub fn set_persistent(&mut self, name: &str, value: TensorValue) {
        self.persistent.insert(name.to_string(), value);
    }

    pub fn get_or_init_persistent(
        &mut self,
        name: &str,
        decl: &VarDecl,
        model: &ModelLoader,
    ) -> Result<TensorValue> {
        if let Some(value) = self.persistent.get(name) {
            return Ok(value.clone());
        }
        let shape = if decl.has_auto_dim() {
            let state = self
                .auto_dims
                .get(name)
                .ok_or_else(|| anyhow!("missing auto_dim state for {}", name))?;
            auto_dim_shape(&state.base_shape, &state.counts)?
        } else {
            model.resolve_shape(&decl.dims)?
        };
        let value = if let Some(init) = &decl.init {
            init.to_tensor_value(decl.dtype, &shape)?
        } else {
            TensorValue::zeros(decl.dtype, &shape)
        };
        self.persistent.insert(name.to_string(), value.clone());
        Ok(value)
    }

    pub fn read(
        &mut self,
        access: &CacheAccess,
        decl: &VarDecl,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
    ) -> Result<TensorValue> {
        if decl.is_cache_table() {
            return self.read_cache_table(access, decl, graph, model, loop_vars);
        }
        self.read_cache_value(access, decl, graph, model, loop_vars)
    }

    pub fn write(
        &mut self,
        src: &TensorValue,
        access: &CacheAccess,
        decl: &VarDecl,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
    ) -> Result<()> {
        if decl.is_cache_table() {
            return self.write_cache_table(src, access, decl, graph, model, loop_vars);
        }
        self.write_cache_value(src, access, decl, graph, model, loop_vars)
    }

    pub fn reset(
        &mut self,
        access: &CacheAccess,
        decl: &VarDecl,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
    ) -> Result<()> {
        if decl.is_cache_table() {
            return self.reset_cache_table(access, decl, graph, model, loop_vars);
        }
        if decl.has_auto_dim() {
            if let Some(state) = self.auto_dims.get_mut(&access.base) {
                state.counts.iter_mut().for_each(|count| *count = 0);
                let shape = auto_dim_shape(&state.base_shape, &state.counts)?;
                self.ensure_persistent_shape(&access.base, decl, &shape)?;
            }
        }
        let shape = model.resolve_shape(&decl.dims)?;
        let value = if let Some(init) = decl.init.as_ref() {
            init.to_tensor_value(decl.dtype, &shape)?
        } else {
            TensorValue::zeros(decl.dtype, &shape)
        };
        self.set_persistent(&access.base, value);
        Ok(())
    }

    pub fn bump(
        &mut self,
        target: &str,
        amount: i64,
        decrement: bool,
        graph: &Graph,
        model: &ModelLoader,
    ) -> Result<()> {
        let decl = graph
            .vars
            .get(target)
            .ok_or_else(|| anyhow!("unknown cache variable: {}", target))?;
        if decl.kind != MemoryKind::Persistent {
            return Err(anyhow!("cache increment expects persistent variable {}", target));
        }
        let value = self.get_or_init_persistent(target, decl, model)?;
        let updated = increment_scalar(value, amount, decrement)?;
        self.set_persistent(target, updated);
        Ok(())
    }

    fn read_cache_value(
        &mut self,
        access: &CacheAccess,
        decl: &VarDecl,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
    ) -> Result<TensorValue> {
        if access.bracketed && !decl.has_auto_dim() {
            return Err(anyhow!(
                "cache {} does not support indexed access",
                access.base
            ));
        }
        if decl.has_auto_dim() {
            if access.bracketed
                && !access.indices.is_empty()
                && access.indices.iter().all(cache_index_is_single)
            {
                self.update_auto_dim_counts_from_access(access, decl, graph, model, loop_vars)?;
                let host = self.get_or_init_persistent(&access.base, decl, model)?;
                return Ok(host);
            }
            let state = self
                .auto_dims
                .get(&access.base)
                .ok_or_else(|| anyhow!("missing auto_dim state for {}", access.base))?;
            let shape = auto_dim_shape(&state.base_shape, &state.counts)?;
            self.ensure_persistent_shape(&access.base, decl, &shape)?;
        }
        let host = self.get_or_init_persistent(&access.base, decl, model)?;
        if !access.bracketed || access.indices.is_empty() {
            return Ok(host);
        }
        let selections = if decl.has_auto_dim() {
            let mut auto_dim_values = HashMap::new();
            if let Some(state) = self.auto_dims.get(&access.base) {
                for (idx, name) in decl.auto_dim.iter().enumerate() {
                    auto_dim_values.insert(name.clone(), state.counts[idx] as i64);
                }
            }
            self.resolve_tensor_indices(&access.indices, host.shape(), graph, model, loop_vars, Some(&auto_dim_values))?
        } else {
            self.resolve_tensor_indices(&access.indices, host.shape(), graph, model, loop_vars, None)?
        };
        slice_tensor_value(&host, &selections)
    }

    fn write_cache_value(
        &mut self,
        src: &TensorValue,
        dst: &CacheAccess,
        decl: &VarDecl,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
    ) -> Result<()> {
        if dst.bracketed && !dst.indices.is_empty() && !decl.has_auto_dim() {
            return Err(anyhow!(
                "cache.write {} does not support indexed writes",
                dst.base
            ));
        }
        if decl.has_auto_dim() {
            if dst.bracketed && !dst.indices.is_empty() {
                if !dst.indices.iter().all(cache_index_is_single) {
                    return Err(anyhow!(
                        "cache.write {} does not support slice indices",
                        dst.base
                    ));
                }
                self.update_auto_dim_counts_from_access(dst, decl, graph, model, loop_vars)?;
            }
            let state = self
                .auto_dims
                .get(&dst.base)
                .ok_or_else(|| anyhow!("missing auto_dim state for {}", dst.base))?;
            let shape = auto_dim_shape(&state.base_shape, &state.counts)?;
            if src.shape() != shape.as_slice() {
                return Err(anyhow!(
                    "cache.write {} expects shape {:?}, got {:?}",
                    dst.base,
                    shape,
                    src.shape()
                ));
            }
        }
        self.set_persistent(&dst.base, src.clone());
        Ok(())
    }

    fn read_cache_table(
        &mut self,
        access: &CacheAccess,
        decl: &VarDecl,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
    ) -> Result<TensorValue> {
        let resolved = self.resolve_cache_index_exprs(access, graph, model, loop_vars)?;
        let entry_shape = self.cache_entry_shape(decl, model)?;
        let table = self
            .tables
            .get_mut(&access.base)
            .ok_or_else(|| anyhow!("missing cache table {}", access.base))?;
        let selections = resolve_table_indices_from_resolved(table, access, &resolved)?;
        read_table_selection(table, &selections, &entry_shape, decl.init.as_ref())
    }

    fn write_cache_table(
        &mut self,
        src: &TensorValue,
        dst: &CacheAccess,
        decl: &VarDecl,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
    ) -> Result<()> {
        let resolved = self.resolve_cache_index_exprs(dst, graph, model, loop_vars)?;
        let entry_shape = self.cache_entry_shape(decl, model)?;
        let table = self
            .tables
            .get_mut(&dst.base)
            .ok_or_else(|| anyhow!("missing cache table {}", dst.base))?;
        let selections = resolve_table_indices_from_resolved(table, dst, &resolved)?;
        if selections.iter().any(|sel| !sel.is_scalar) {
            return Err(anyhow!(
                "cache.write {} does not support slice indices",
                dst.base
            ));
        }
        let mut indices = Vec::with_capacity(selections.len());
        for selection in selections {
            indices.push(*selection.indices.first().unwrap_or(&0));
        }
        if src.shape() != entry_shape.as_slice() {
            return Err(anyhow!(
                "cache.write {} expects shape {:?}, got {:?}",
                dst.base,
                entry_shape,
                src.shape()
            ));
        }
        table.entries.insert(indices, src.clone());
        Ok(())
    }

    fn reset_cache_table(
        &mut self,
        target: &CacheAccess,
        decl: &VarDecl,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
    ) -> Result<()> {
        let resolved = self.resolve_cache_index_exprs(target, graph, model, loop_vars)?;
        let table = self
            .tables
            .get_mut(&target.base)
            .ok_or_else(|| anyhow!("missing cache table {}", target.base))?;
        if !target.bracketed || target.indices.is_empty() {
            table.entries.clear();
            reset_cache_table_sizes(table, decl)?;
            return Ok(());
        }
        let prefixes = resolve_table_prefix_indices_from_resolved(target, &resolved)?;
        table.entries.retain(|indices, _| {
            if indices.len() < prefixes.len() {
                return true;
            }
            for (idx, prefix) in prefixes.iter().enumerate() {
                if indices[idx] != *prefix {
                    return true;
                }
            }
            false
        });
        recompute_cache_table_sizes(table, decl)?;
        Ok(())
    }

    fn cache_entry_shape(&self, decl: &VarDecl, model: &ModelLoader) -> Result<Vec<usize>> {
        let mut shape = model.resolve_shape(&decl.dims)?;
        if decl.has_auto_dim() {
            if let Some(state) = self.auto_dims.get(&decl.name) {
                shape = auto_dim_shape(&state.base_shape, &state.counts)?;
            }
        }
        Ok(shape)
    }

    fn ensure_persistent_shape(
        &mut self,
        name: &str,
        decl: &VarDecl,
        shape: &[usize],
    ) -> Result<()> {
        let current = self
            .persistent
            .get(name)
            .cloned()
            .unwrap_or_else(|| TensorValue::zeros(decl.dtype, shape));
        if current.shape() == shape {
            self.set_persistent(name, current);
            return Ok(());
        }
        let resized = expand_tensor_value(&current, shape)?;
        if resized.dtype() != decl.dtype {
            return Err(anyhow!(
                "cache {} dtype mismatch {:?} vs {:?}",
                name,
                resized.dtype(),
                decl.dtype
            ));
        }
        self.set_persistent(name, resized);
        Ok(())
    }

    fn resolve_cache_index_exprs(
        &mut self,
        access: &CacheAccess,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
    ) -> Result<Vec<ResolvedCacheIndexExpr>> {
        let mut resolved = Vec::with_capacity(access.indices.len());
        for expr in &access.indices {
            let value = match expr {
                CacheIndexExpr::Single(value) => {
                    ResolvedCacheIndexExpr::Single(
                        self.resolve_cache_index_value_usize(value, graph, model, loop_vars)?,
                    )
                }
                CacheIndexExpr::Slice { start, end } => {
                    let start = match start {
                        Some(value) => Some(self.resolve_cache_index_value(value, graph, model, loop_vars)?),
                        None => None,
                    };
                    let end = match end {
                        Some(value) => Some(self.resolve_cache_index_value(value, graph, model, loop_vars)?),
                        None => None,
                    };
                    ResolvedCacheIndexExpr::Slice { start, end }
                }
            };
            resolved.push(value);
        }
        Ok(resolved)
    }

    fn resolve_tensor_indices(
        &mut self,
        indices: &[CacheIndexExpr],
        shape: &[usize],
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
        specials: Option<&HashMap<String, i64>>,
    ) -> Result<Vec<TensorIndexSelection>> {
        if !indices.is_empty() && indices.len() > shape.len() {
            return Err(anyhow!(
                "cache access expects at most {} indices, got {}",
                shape.len(),
                indices.len()
            ));
        }
        let mut selections = Vec::with_capacity(shape.len());
        for dim in 0..shape.len() {
            let expr = indices.get(dim);
            let selection = match expr {
                None => TensorIndexSelection {
                    indices: (0..shape[dim]).collect(),
                    is_scalar: false,
                },
                Some(CacheIndexExpr::Single(value)) => {
                    let index = self.resolve_cache_index_value_usize_with_map(
                        value,
                        graph,
                        model,
                        loop_vars,
                        specials,
                    )?;
                    if index >= shape[dim] {
                        return Err(anyhow!(
                            "cache index {} out of bounds for dim {} (size {})",
                            index,
                            dim,
                            shape[dim]
                        ));
                    }
                    TensorIndexSelection {
                        indices: vec![index],
                        is_scalar: true,
                    }
                }
                Some(CacheIndexExpr::Slice { start, end }) => {
                    let start = match start {
                        Some(value) => Some(self.resolve_cache_index_value_with_map(
                            value,
                            graph,
                            model,
                            loop_vars,
                            specials,
                        )?),
                        None => None,
                    };
                    let end = match end {
                        Some(value) => Some(self.resolve_cache_index_value_with_map(
                            value,
                            graph,
                            model,
                            loop_vars,
                            specials,
                        )?),
                        None => None,
                    };
                    let (start, end) = resolve_slice_bounds(shape[dim], start, end, true)?;
                    TensorIndexSelection {
                        indices: (start..end).collect(),
                        is_scalar: false,
                    }
                }
            };
            selections.push(selection);
        }
        Ok(selections)
    }

    fn resolve_cache_index_value_with_map(
        &mut self,
        value: &CacheIndexValue,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
        specials: Option<&HashMap<String, i64>>,
    ) -> Result<i64> {
        match value {
            CacheIndexValue::Lit(value) => Ok(*value),
            CacheIndexValue::Ident(name) => {
                if let Ok(value) = model.size_of(name) {
                    return Ok(value as i64);
                }
                if let Some(map) = specials {
                    if let Some(value) = map.get(name) {
                        return Ok(*value);
                    }
                }
                if let Some(value) = loop_vars.get(name) {
                    return Ok(*value);
                }
                if let Some(decl) = graph.vars.get(name) {
                    if decl.kind == MemoryKind::Persistent {
                        let tensor = self.get_or_init_persistent(name, decl, model)?;
                        return scalar_to_i64(&tensor);
                    }
                }
                Err(anyhow!("unknown cache index {}", name))
            }
        }
    }

    fn resolve_cache_index_value(
        &mut self,
        value: &CacheIndexValue,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
    ) -> Result<i64> {
        self.resolve_cache_index_value_with_map(value, graph, model, loop_vars, None)
    }

    fn resolve_cache_index_value_usize_with_map(
        &mut self,
        value: &CacheIndexValue,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
        specials: Option<&HashMap<String, i64>>,
    ) -> Result<usize> {
        let value = self.resolve_cache_index_value_with_map(value, graph, model, loop_vars, specials)?;
        if value < 0 {
            return Err(anyhow!("cache index cannot be negative"));
        }
        Ok(value as usize)
    }

    fn resolve_cache_index_value_usize(
        &mut self,
        value: &CacheIndexValue,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
    ) -> Result<usize> {
        self.resolve_cache_index_value_usize_with_map(value, graph, model, loop_vars, None)
    }

    fn update_auto_dim_counts_from_access(
        &mut self,
        access: &CacheAccess,
        decl: &VarDecl,
        graph: &Graph,
        model: &ModelLoader,
        loop_vars: &HashMap<String, i64>,
    ) -> Result<()> {
        let mut requested_values = Vec::new();
        for expr in &access.indices {
            let CacheIndexExpr::Single(value) = expr else {
                return Err(anyhow!(
                    "cache access {} requires scalar indices",
                    access.base
                ));
            };
            requested_values.push(self.resolve_cache_index_value_usize(value, graph, model, loop_vars)?);
        }

        let base_shape;
        let mut counts;
        {
            let state = self
                .auto_dims
                .get_mut(&access.base)
                .ok_or_else(|| anyhow!("missing auto_dim state for {}", access.base))?;
            if requested_values.len() > state.counts.len() {
                return Err(anyhow!(
                    "cache access {} expects at most {} indices, got {}",
                    access.base,
                    state.counts.len(),
                    requested_values.len()
                ));
            }
            base_shape = state.base_shape.clone();
            counts = state.counts.clone();
            for (idx, requested) in requested_values.iter().enumerate() {
                if let Some(limit) = state.max.get(idx).copied().flatten() {
                    if *requested > limit {
                        return Err(anyhow!(
                            "auto_dim {} exceeds fixed max {} for {}",
                            access.base,
                            limit,
                            idx
                        ));
                    }
                }
                if counts[idx] < *requested {
                    counts[idx] = *requested;
                }
            }
            state.counts = counts.clone();
        }
        let shape = auto_dim_shape(&base_shape, &counts)?;
        self.ensure_persistent_shape(&access.base, decl, &shape)
    }
}

fn resolve_slice_bounds(
    size: usize,
    start: Option<i64>,
    end: Option<i64>,
    allow_negative_end: bool,
) -> Result<(usize, usize)> {
    let start = start.unwrap_or(0);
    if start < 0 {
        return Err(anyhow!("slice start cannot be negative"));
    }
    let end = match end {
        Some(value) if value < 0 => {
            if !allow_negative_end {
                return Err(anyhow!("slice end cannot be negative"));
            }
            let abs = value.abs() as usize;
            if abs > size {
                return Err(anyhow!("slice end underflow for size {}", size));
            }
            (size - abs) as i64
        }
        Some(value) => value,
        None => size as i64,
    };
    if end < 0 {
        return Err(anyhow!("slice end cannot be negative"));
    }
    let start = start as usize;
    let end = end as usize;
    if end < start {
        return Err(anyhow!("slice end {} before start {}", end, start));
    }
    Ok((start, end))
}
