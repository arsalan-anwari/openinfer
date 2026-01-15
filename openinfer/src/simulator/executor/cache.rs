use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::graph::{CacheAccess, CacheIndexExpr, CacheIndexValue};
use crate::tensor::TensorValue;
use crate::types::MemoryKind;

use super::tensor_utils::{
    expand_tensor_value, increment_scalar, scalar_to_i64, slice_tensor_value,
};
use super::Executor;

#[derive(Debug, Clone)]
pub(super) struct AutoDimState {
    pub(super) base_shape: Vec<usize>,
    pub(super) counts: Vec<usize>,
    pub(super) max: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
pub(super) struct CacheTable {
    pub(super) decl: crate::types::VarDecl,
    #[allow(dead_code)]
    pub(super) base_shape: Vec<usize>,
    pub(super) table_indices: Vec<String>,
    pub(super) fixed_sizes: Vec<Option<usize>>,
    pub(super) sizes: Vec<usize>,
    pub(super) entries: HashMap<Vec<usize>, TensorValue>,
}

#[derive(Debug, Clone)]
pub(super) struct TableIndexSelection {
    pub(super) indices: Vec<usize>,
    pub(super) is_scalar: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct TensorIndexSelection {
    pub(super) indices: Vec<usize>,
    pub(super) is_scalar: bool,
}

#[derive(Debug, Clone)]
pub(super) enum ResolvedCacheIndexExpr {
    Single(usize),
    Slice { start: Option<i64>, end: Option<i64> },
}

impl Executor<'_> {
    pub(super) fn advance_auto_dims(&mut self) -> Result<()> {
        let names: Vec<String> = self.auto_dims.keys().cloned().collect();
        for name in names {
            let Some(state) = self.auto_dims.get_mut(&name) else {
                continue;
            };
            let shape = auto_dim_shape(&state.base_shape, &state.counts)?;
            if let Some(decl) = self.graph.vars.get(&name) {
                if !decl.is_cache_table() {
                    let decl = decl.clone();
                    self.ensure_persistent_shape(&name, &decl, &shape)?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn exec_cache_read(&mut self, src: &CacheAccess, dst: &str) -> Result<()> {
        let decl = self
            .graph
            .vars
            .get(&src.base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown cache variable: {}", src.base))?;
        if decl.is_cache_table() {
            let value = self.read_cache_table(src, &decl)?;
            return self.write_output_tensor(dst, value);
        }
        let value = self.read_cache_value(src, &decl)?;
        self.write_output_tensor(dst, value)
    }

    pub(super) fn exec_cache_write(&mut self, src: &str, dst: &CacheAccess) -> Result<()> {
        let decl = self
            .graph
            .vars
            .get(&dst.base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown cache variable: {}", dst.base))?;
        if decl.is_cache_table() {
            return self.write_cache_table(src, dst, &decl);
        }
        self.write_cache_value(src, dst, &decl)
    }

    pub(super) fn exec_cache_increment(
        &mut self,
        target: &str,
        amount: i64,
        decrement: bool,
    ) -> Result<()> {
        let decl = self
            .graph
            .vars
            .get(target)
            .ok_or_else(|| anyhow!("unknown cache variable: {}", target))?;
        if decl.kind != MemoryKind::Persistent {
            return Err(anyhow!("cache increment expects persistent variable {}", target));
        }
        let value = self.fetch_persistent_tensor(target)?;
        let updated = increment_scalar(value, amount, decrement)?;
        self.store_persistent_tensor(target, updated)
    }

    pub(super) fn exec_cache_reset(&mut self, target: &CacheAccess) -> Result<()> {
        let decl = self
            .graph
            .vars
            .get(&target.base)
            .cloned()
            .ok_or_else(|| anyhow!("unknown cache variable: {}", target.base))?;
        if decl.is_cache_table() {
            return self.reset_cache_table(target, &decl);
        }
        if decl.has_auto_dim() {
            if let Some(state) = self.auto_dims.get_mut(&target.base) {
                state.counts.iter_mut().for_each(|count| *count = 0);
                let shape = auto_dim_shape(&state.base_shape, &state.counts)?;
                self.ensure_persistent_shape(&target.base, &decl, &shape)?;
            }
        }
        let shape = self.model.resolve_shape(&decl.dims)?;
        let value = if let Some(init) = decl.init.as_ref() {
            init.to_tensor_value(decl.dtype, &shape)?
        } else {
            TensorValue::zeros(decl.dtype, &shape)
        };
        self.store_persistent_tensor(&target.base, value)
    }

    pub(super) fn read_cache_value(
        &mut self,
        access: &CacheAccess,
        decl: &crate::types::VarDecl,
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
                && access.indices.iter().all(is_cache_index_single)
            {
                self.update_auto_dim_counts_from_access(access, decl)?;
                let host = self.fetch_persistent_tensor(&access.base)?;
                return Ok(host);
            }
            let state = self
                .auto_dims
                .get(&access.base)
                .ok_or_else(|| anyhow!("missing auto_dim state for {}", access.base))?;
            let shape = auto_dim_shape(&state.base_shape, &state.counts)?;
            self.ensure_persistent_shape(&access.base, decl, &shape)?;
        }
        let host = self.fetch_persistent_tensor(&access.base)?;
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
            self.resolve_tensor_indices(&access.indices, host.shape(), Some(&auto_dim_values))?
        } else {
            self.resolve_tensor_indices(&access.indices, host.shape(), None)?
        };
        slice_tensor_value(&host, &selections)
    }

    pub(super) fn write_cache_value(
        &mut self,
        src: &str,
        dst: &CacheAccess,
        decl: &crate::types::VarDecl,
    ) -> Result<()> {
        if dst.bracketed && !dst.indices.is_empty() && !decl.has_auto_dim() {
            return Err(anyhow!(
                "cache.write {} does not support indexed writes",
                dst.base
            ));
        }
        let storage = self.get_tensor(src)?;
        let input = self.backend.download(storage)?;
        if decl.has_auto_dim() {
            if dst.bracketed && !dst.indices.is_empty() {
                if !dst.indices.iter().all(is_cache_index_single) {
                    return Err(anyhow!(
                        "cache.write {} does not support slice indices",
                        dst.base
                    ));
                }
                self.update_auto_dim_counts_from_access(dst, decl)?;
            }
            let state = self
                .auto_dims
                .get(&dst.base)
                .ok_or_else(|| anyhow!("missing auto_dim state for {}", dst.base))?;
            let shape = auto_dim_shape(&state.base_shape, &state.counts)?;
            if input.shape() != shape.as_slice() {
                return Err(anyhow!(
                    "cache.write {} expects shape {:?}, got {:?}",
                    dst.base,
                    shape,
                    input.shape()
                ));
            }
        }
        self.store_persistent_tensor(&dst.base, input)
    }

    pub(super) fn read_cache_table(
        &mut self,
        access: &CacheAccess,
        decl: &crate::types::VarDecl,
    ) -> Result<TensorValue> {
        let resolved = self.resolve_cache_index_exprs(access)?;
        let entry_shape = self.cache_entry_shape(decl)?;
        let table = self
            .cache_tables
            .get_mut(&access.base)
            .ok_or_else(|| anyhow!("missing cache table {}", access.base))?;
        let selections = resolve_table_indices_from_resolved(table, access, &resolved)?;
        read_table_selection(table, &selections, &entry_shape, decl.init.as_ref())
    }

    pub(super) fn write_cache_table(
        &mut self,
        src: &str,
        dst: &CacheAccess,
        decl: &crate::types::VarDecl,
    ) -> Result<()> {
        let resolved = self.resolve_cache_index_exprs(dst)?;
        let storage = self.get_tensor(src)?;
        let input = self.backend.download(storage)?;
        let entry_shape = self.cache_entry_shape(decl)?;
        let table = self
            .cache_tables
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
        if input.shape() != entry_shape.as_slice() {
            return Err(anyhow!(
                "cache.write {} expects shape {:?}, got {:?}",
                dst.base,
                entry_shape,
                input.shape()
            ));
        }
        table.entries.insert(indices, input);
        Ok(())
    }

    pub(super) fn reset_cache_table(
        &mut self,
        target: &CacheAccess,
        decl: &crate::types::VarDecl,
    ) -> Result<()> {
        let resolved = self.resolve_cache_index_exprs(target)?;
        let table = self
            .cache_tables
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

    pub(super) fn cache_entry_shape(&self, decl: &crate::types::VarDecl) -> Result<Vec<usize>> {
        let mut shape = self.model.resolve_shape(&decl.dims)?;
        if decl.has_auto_dim() {
            if let Some(state) = self.auto_dims.get(&decl.name) {
                shape = auto_dim_shape(&state.base_shape, &state.counts)?;
            }
        }
        Ok(shape)
    }

    pub(super) fn ensure_persistent_shape(
        &mut self,
        name: &str,
        decl: &crate::types::VarDecl,
        shape: &[usize],
    ) -> Result<()> {
        let current = self.fetch_persistent_tensor(name)?;
        if current.shape() == shape {
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
        self.store_persistent_tensor(name, resized)
    }

    pub(super) fn fetch_persistent_tensor(&mut self, name: &str) -> Result<TensorValue> {
        let storage = self.get_tensor(name)?;
        self.backend.download(storage)
    }

    pub(super) fn store_persistent_tensor(
        &mut self,
        name: &str,
        value: TensorValue,
    ) -> Result<()> {
        let uploaded = self.backend.upload(value)?;
        self.storage
            .insert(name.to_string(), super::StoredTensor::Data(uploaded));
        Ok(())
    }

    pub(super) fn write_output_tensor(&mut self, name: &str, value: TensorValue) -> Result<()> {
        if self.kinds.get(name) == Some(&MemoryKind::Dynamic) {
            self.dynamic
                .insert(name.to_string(), self.backend.upload(value)?);
        } else {
            self.storage.insert(
                name.to_string(),
                super::StoredTensor::Data(self.backend.upload(value)?),
            );
        }
        Ok(())
    }

    pub(super) fn resolve_cache_index_exprs(
        &mut self,
        access: &CacheAccess,
    ) -> Result<Vec<ResolvedCacheIndexExpr>> {
        let mut resolved = Vec::with_capacity(access.indices.len());
        for expr in &access.indices {
            let value = match expr {
                CacheIndexExpr::Single(value) => {
                    ResolvedCacheIndexExpr::Single(self.resolve_cache_index_value_usize(value)?)
                }
                CacheIndexExpr::Slice { start, end } => {
                    let start = match start {
                        Some(value) => Some(self.resolve_cache_index_value(value)?),
                        None => None,
                    };
                    let end = match end {
                        Some(value) => Some(self.resolve_cache_index_value(value)?),
                        None => None,
                    };
                    ResolvedCacheIndexExpr::Slice { start, end }
                }
            };
            resolved.push(value);
        }
        Ok(resolved)
    }

    pub(super) fn resolve_tensor_indices(
        &mut self,
        indices: &[CacheIndexExpr],
        shape: &[usize],
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
                    let index = self.resolve_cache_index_value_usize_with_map(value, specials)?;
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
                        Some(value) => Some(self.resolve_cache_index_value_with_map(value, specials)?),
                        None => None,
                    };
                    let end = match end {
                        Some(value) => Some(self.resolve_cache_index_value_with_map(value, specials)?),
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

    pub(super) fn resolve_cache_index_value_with_map(
        &mut self,
        value: &CacheIndexValue,
        specials: Option<&HashMap<String, i64>>,
    ) -> Result<i64> {
        match value {
            CacheIndexValue::Lit(value) => Ok(*value),
            CacheIndexValue::Ident(name) => {
                if let Some(map) = specials {
                    if let Some(value) = map.get(name) {
                        return Ok(*value);
                    }
                }
                if let Some(value) = self.loop_vars.get(name) {
                    return Ok(*value as i64);
                }
                if self.kinds.get(name) == Some(&MemoryKind::Persistent) {
                    let tensor = self.fetch_persistent_tensor(name)?;
                    return scalar_to_i64(&tensor);
                }
                Err(anyhow!("unknown cache index {}", name))
            }
        }
    }

    pub(super) fn resolve_cache_index_value(&mut self, value: &CacheIndexValue) -> Result<i64> {
        self.resolve_cache_index_value_with_map(value, None)
    }

    pub(super) fn resolve_cache_index_value_usize_with_map(
        &mut self,
        value: &CacheIndexValue,
        specials: Option<&HashMap<String, i64>>,
    ) -> Result<usize> {
        let value = self.resolve_cache_index_value_with_map(value, specials)?;
        if value < 0 {
            return Err(anyhow!("cache index cannot be negative"));
        }
        Ok(value as usize)
    }

    pub(super) fn resolve_cache_index_value_usize(
        &mut self,
        value: &CacheIndexValue,
    ) -> Result<usize> {
        self.resolve_cache_index_value_usize_with_map(value, None)
    }

    pub(super) fn update_auto_dim_counts_from_access(
        &mut self,
        access: &CacheAccess,
        decl: &crate::types::VarDecl,
    ) -> Result<()> {
        let mut requested_values = Vec::new();
        for expr in &access.indices {
            let CacheIndexExpr::Single(value) = expr else {
                return Err(anyhow!(
                    "cache access {} requires scalar indices",
                    access.base
                ));
            };
            requested_values.push(self.resolve_cache_index_value_usize(value)?);
        }

        #[allow(unused_mut)]
        let mut base_shape;
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

pub(super) fn resolve_table_indices_from_resolved(
    table: &mut CacheTable,
    access: &CacheAccess,
    resolved: &[ResolvedCacheIndexExpr],
) -> Result<Vec<TableIndexSelection>> {
    if !access.bracketed {
        return Err(anyhow!("cache table {} requires indices", access.base));
    }
    let rank = table.table_indices.len();
    if !resolved.is_empty() && resolved.len() > rank {
        return Err(anyhow!(
            "cache table {} expects {} indices, got {}",
            access.base,
            rank,
            resolved.len()
        ));
    }
    let mut selections = Vec::with_capacity(rank);
    for dim in 0..rank {
        let expr = resolved.get(dim);
        let selection = match expr {
            None => resolve_table_slice(table, dim, None, None)?,
            Some(ResolvedCacheIndexExpr::Single(index)) => {
                ensure_table_size(table, dim, index + 1)?;
                TableIndexSelection {
                    indices: vec![*index],
                    is_scalar: true,
                }
            }
            Some(ResolvedCacheIndexExpr::Slice { start, end }) => {
                resolve_table_slice(table, dim, *start, *end)?
            }
        };
        selections.push(selection);
    }
    Ok(selections)
}

fn resolve_table_prefix_indices_from_resolved(
    access: &CacheAccess,
    resolved: &[ResolvedCacheIndexExpr],
) -> Result<Vec<usize>> {
    if !access.bracketed {
        return Err(anyhow!("cache table {} requires indices", access.base));
    }
    let mut prefixes = Vec::new();
    for expr in resolved {
        match expr {
            ResolvedCacheIndexExpr::Single(value) => prefixes.push(*value),
            ResolvedCacheIndexExpr::Slice { .. } => {
                return Err(anyhow!(
                    "cache.reset {} requires scalar indices",
                    access.base
                ));
            }
        }
    }
    Ok(prefixes)
}

pub(super) fn fixed_size_for(decl: &crate::types::VarDecl, index: &str) -> Option<usize> {
    decl.fixed
        .iter()
        .find(|(name, _)| name == index)
        .map(|(_, value)| *value)
}

pub(super) fn init_cache_table_sizes(
    decl: &crate::types::VarDecl,
    table_indices: &[String],
) -> Result<(Vec<Option<usize>>, Vec<usize>)> {
    let mut fixed_sizes = Vec::with_capacity(table_indices.len());
    let mut sizes = Vec::with_capacity(table_indices.len());
    for index in table_indices {
        let fixed = fixed_size_for(decl, index);
        fixed_sizes.push(fixed);
        sizes.push(fixed.unwrap_or(0));
    }
    Ok((fixed_sizes, sizes))
}

fn reset_cache_table_sizes(table: &mut CacheTable, decl: &crate::types::VarDecl) -> Result<()> {
    let (fixed_sizes, sizes) = init_cache_table_sizes(decl, &table.table_indices)?;
    table.fixed_sizes = fixed_sizes;
    table.sizes = sizes;
    Ok(())
}

fn recompute_cache_table_sizes(table: &mut CacheTable, decl: &crate::types::VarDecl) -> Result<()> {
    let mut sizes = vec![0usize; table.table_indices.len()];
    for indices in table.entries.keys() {
        for (dim, value) in indices.iter().enumerate() {
            sizes[dim] = sizes[dim].max(value.saturating_add(1));
        }
    }
    for (dim, index) in table.table_indices.iter().enumerate() {
        if let Some(fixed) = fixed_size_for(decl, index) {
            sizes[dim] = fixed;
        }
    }
    table.sizes = sizes;
    Ok(())
}

pub(super) fn auto_dim_shape(base: &[usize], counts: &[usize]) -> Result<Vec<usize>> {
    if base.len() != counts.len() {
        return Err(anyhow!(
            "auto_dim requires {} dims, got {}",
            counts.len(),
            base.len()
        ));
    }
    Ok(base
        .iter()
        .zip(counts.iter())
        .map(|(base, count)| base.saturating_add(*count))
        .collect())
}

fn ensure_table_size(table: &mut CacheTable, dim: usize, required: usize) -> Result<()> {
    if let Some(max) = table.fixed_sizes.get(dim).copied().flatten() {
        if required > max {
            return Err(anyhow!(
                "cache table {} index {} exceeds fixed size {}",
                table.decl.name,
                dim,
                max
            ));
        }
    }
    if table.sizes[dim] < required {
        table.sizes[dim] = required;
    }
    Ok(())
}

fn resolve_table_slice(
    table: &mut CacheTable,
    dim: usize,
    start: Option<i64>,
    end: Option<i64>,
) -> Result<TableIndexSelection> {
    let size = table.sizes[dim];
    let (start, end) = resolve_slice_bounds(size, start, end, true)?;
    ensure_table_size(table, dim, end)?;
    Ok(TableIndexSelection {
        indices: (start..end).collect(),
        is_scalar: false,
    })
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

fn read_table_selection(
    table: &mut CacheTable,
    selections: &[TableIndexSelection],
    entry_shape: &[usize],
    init: Option<&crate::types::ScalarValue>,
) -> Result<TensorValue> {
    let entry_len = crate::tensor::numel(entry_shape);
    let mut output_shape = Vec::new();
    for selection in selections {
        if !selection.is_scalar {
            output_shape.push(selection.indices.len());
        }
    }
    output_shape.extend_from_slice(entry_shape);
    match table.decl.dtype {
        crate::tensor::DType::I8 => {
            let shape = output_shape.clone();
            read_table_values_i8(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::I8(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::I16 => {
            let shape = output_shape.clone();
            read_table_values_i16(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::I16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::I32 => {
            let shape = output_shape.clone();
            read_table_values_i32(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::I32(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::I64 => {
            let shape = output_shape.clone();
            read_table_values_i64(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::I64(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::U8 => {
            let shape = output_shape.clone();
            read_table_values_u8(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::U8(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::U16 => {
            let shape = output_shape.clone();
            read_table_values_u16(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::U16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::U32 => {
            let shape = output_shape.clone();
            read_table_values_u32(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::U32(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::U64 => {
            let shape = output_shape.clone();
            read_table_values_u64(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::U64(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::F16 => {
            let shape = output_shape.clone();
            read_table_values_f16(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::F16(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::F32 => {
            let shape = output_shape.clone();
            read_table_values_f32(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::F32(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::F64 => {
            let shape = output_shape.clone();
            read_table_values_f64(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::F64(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::Bool => {
            let shape = output_shape.clone();
            read_table_values_bool(table, selections, entry_shape, entry_len, init).and_then(|data| {
                Ok(TensorValue::Bool(crate::tensor::Tensor::from_vec_with_opts(
                    data,
                    crate::tensor::TensorOptions {
                        shape: Some(shape),
                        ..crate::tensor::TensorOptions::default()
                    },
                )?))
            })
        }
        crate::tensor::DType::Bitset => {
            let shape = output_shape.clone();
            read_table_values_bitset(table, selections, entry_shape, entry_len, init)
                .and_then(|data| {
                    Ok(TensorValue::Bitset(crate::tensor::Tensor::from_vec_with_opts(
                        data,
                        crate::tensor::TensorOptions {
                            shape: Some(shape),
                            ..crate::tensor::TensorOptions::default()
                        },
                    )?))
                })
        }
    }
}

macro_rules! read_table_values {
    ($read_name:ident, $entry_name:ident, $variant:ident, $ty:ty) => {
        fn $read_name(
            table: &mut CacheTable,
            selections: &[TableIndexSelection],
            entry_shape: &[usize],
            entry_len: usize,
            init: Option<&crate::types::ScalarValue>,
        ) -> Result<Vec<$ty>> {
            let mut output = Vec::new();
            let mut current = vec![0usize; selections.len()];
            fn recurse(
                table: &mut CacheTable,
                selections: &[TableIndexSelection],
                entry_shape: &[usize],
                entry_len: usize,
                init: Option<&crate::types::ScalarValue>,
                depth: usize,
                current: &mut [usize],
                output: &mut Vec<$ty>,
            ) -> Result<()> {
                if depth == selections.len() {
                    let entry = $entry_name(table, current, entry_shape, entry_len, init)?;
                    output.extend_from_slice(&entry);
                    return Ok(());
                }
                for index in &selections[depth].indices {
                    current[depth] = *index;
                    recurse(
                        table,
                        selections,
                        entry_shape,
                        entry_len,
                        init,
                        depth + 1,
                        current,
                        output,
                    )?;
                }
                Ok(())
            }
            recurse(
                table,
                selections,
                entry_shape,
                entry_len,
                init,
                0,
                &mut current,
                &mut output,
            )?;
            Ok(output)
        }

        fn $entry_name(
            table: &mut CacheTable,
            indices: &[usize],
            entry_shape: &[usize],
            entry_len: usize,
            init: Option<&crate::types::ScalarValue>,
        ) -> Result<Vec<$ty>> {
            if let Some(entry) = table.entries.get(indices) {
                if let TensorValue::$variant(t) = entry {
                    if t.shape() != entry_shape {
                        let resized = expand_tensor_value(entry, entry_shape)?;
                        table.entries.insert(indices.to_vec(), resized);
                    }
                    if let TensorValue::$variant(t) = table.entries.get(indices).unwrap() {
                        return Ok(t.data.clone());
                    }
                    return Err(anyhow!("cache table entry dtype mismatch"));
                }
                return Err(anyhow!("cache table entry dtype mismatch"));
            }
            let value = if let Some(init) = init {
                init.to_tensor_value(table.decl.dtype, entry_shape)?
            } else {
                TensorValue::zeros(table.decl.dtype, entry_shape)
            };
            let entry = if let TensorValue::$variant(t) = &value {
                t.data.clone()
            } else {
                return Err(anyhow!("cache table entry dtype mismatch"));
            };
            if entry.len() != entry_len {
                return Err(anyhow!("cache table entry length mismatch"));
            }
            table.entries.insert(indices.to_vec(), value);
            Ok(entry)
        }
    };
}

read_table_values!(read_table_values_i8, get_table_entry_i8, I8, i8);
read_table_values!(read_table_values_i16, get_table_entry_i16, I16, i16);
read_table_values!(read_table_values_i32, get_table_entry_i32, I32, i32);
read_table_values!(read_table_values_i64, get_table_entry_i64, I64, i64);
read_table_values!(read_table_values_u8, get_table_entry_u8, U8, u8);
read_table_values!(read_table_values_u16, get_table_entry_u16, U16, u16);
read_table_values!(read_table_values_u32, get_table_entry_u32, U32, u32);
read_table_values!(read_table_values_u64, get_table_entry_u64, U64, u64);
read_table_values!(
    read_table_values_f16,
    get_table_entry_f16,
    F16,
    crate::tensor::F16
);
read_table_values!(read_table_values_f32, get_table_entry_f32, F32, f32);
read_table_values!(read_table_values_f64, get_table_entry_f64, F64, f64);
read_table_values!(read_table_values_bool, get_table_entry_bool, Bool, bool);
read_table_values!(
    read_table_values_bitset,
    get_table_entry_bitset,
    Bitset,
    crate::tensor::Bitset
);

pub(crate) fn format_cache_access(access: &CacheAccess) -> String {
    if !access.bracketed {
        return access.base.clone();
    }
    if access.indices.is_empty() {
        return format!("{}[]", access.base);
    }
    let rendered = access
        .indices
        .iter()
        .map(|index| match index {
            CacheIndexExpr::Single(value) => format_cache_value(value),
            CacheIndexExpr::Slice { start, end } => {
                let start = start.as_ref().map(format_cache_value);
                let end = end.as_ref().map(format_cache_value);
                match (start, end) {
                    (Some(start), Some(end)) => format!("{}..{}", start, end),
                    (Some(start), None) => format!("{}..", start),
                    (None, Some(end)) => format!("..{}", end),
                    (None, None) => String::new(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{}[{}]", access.base, rendered)
}

fn format_cache_value(value: &CacheIndexValue) -> String {
    match value {
        CacheIndexValue::Ident(name) => name.clone(),
        CacheIndexValue::Lit(value) => value.to_string(),
    }
}

fn is_cache_index_single(expr: &CacheIndexExpr) -> bool {
    matches!(expr, CacheIndexExpr::Single(_))
}
