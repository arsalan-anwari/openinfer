use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::graph::{CacheAccess, VarDecl};
use crate::tensor::{DType, TensorValue};

use super::auto_dim::{fixed_size_for, init_cache_table_sizes};
use super::{ResolvedCacheIndexExpr, TableIndexSelection};

#[derive(Debug, Clone)]
pub struct CacheTable {
    pub decl: VarDecl,
    pub table_indices: Vec<String>,
    pub fixed_sizes: Vec<Option<usize>>,
    pub sizes: Vec<usize>,
    pub entries: HashMap<Vec<usize>, TensorValue>,
}

pub fn build_cache_table(decl: &VarDecl) -> Result<CacheTable> {
    let table_indices = decl.cache_table_indices();
    let (fixed_sizes, sizes) = init_cache_table_sizes(decl, &table_indices)?;
    Ok(CacheTable {
        decl: decl.clone(),
        table_indices,
        fixed_sizes,
        sizes,
        entries: HashMap::new(),
    })
}

pub fn resolve_table_indices_from_resolved(
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

pub fn resolve_table_prefix_indices_from_resolved(
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

pub fn reset_cache_table_sizes(table: &mut CacheTable, decl: &VarDecl) -> Result<()> {
    let (fixed_sizes, sizes) = init_cache_table_sizes(decl, &table.table_indices)?;
    table.fixed_sizes = fixed_sizes;
    table.sizes = sizes;
    Ok(())
}

pub fn recompute_cache_table_sizes(table: &mut CacheTable, decl: &VarDecl) -> Result<()> {
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

pub fn read_table_selection(
    table: &mut CacheTable,
    selections: &[TableIndexSelection],
    entry_shape: &[usize],
    init: Option<&crate::types::ScalarValue>,
) -> Result<TensorValue> {
    let logical_len = crate::tensor::numel(entry_shape);
    let entry_len = table.decl.dtype.storage_len(logical_len);
    let mut output_shape = Vec::new();
    for selection in selections {
        if !selection.is_scalar {
            output_shape.push(selection.indices.len());
        }
    }
    output_shape.extend_from_slice(entry_shape);
    match table.decl.dtype {
        DType::I8 => read_table_values::<i8, _>(table, selections, entry_shape, entry_len, init, output_shape, TensorValue::I8),
        DType::I16 => read_table_values::<i16, _>(table, selections, entry_shape, entry_len, init, output_shape, TensorValue::I16),
        DType::I32 => read_table_values::<i32, _>(table, selections, entry_shape, entry_len, init, output_shape, TensorValue::I32),
        DType::I64 => read_table_values::<i64, _>(table, selections, entry_shape, entry_len, init, output_shape, TensorValue::I64),
        DType::U8 => read_table_values::<u8, _>(table, selections, entry_shape, entry_len, init, output_shape, TensorValue::U8),
        DType::U16 => read_table_values::<u16, _>(table, selections, entry_shape, entry_len, init, output_shape, TensorValue::U16),
        DType::U32 => read_table_values::<u32, _>(table, selections, entry_shape, entry_len, init, output_shape, TensorValue::U32),
        DType::U64 => read_table_values::<u64, _>(table, selections, entry_shape, entry_len, init, output_shape, TensorValue::U64),
        DType::F16 => read_table_values::<crate::tensor::F16, _>(
            table,
            selections,
            entry_shape,
            entry_len,
            init,
            output_shape,
            TensorValue::F16,
        ),
        DType::BF16 => read_table_values::<crate::tensor::BF16, _>(
            table,
            selections,
            entry_shape,
            entry_len,
            init,
            output_shape,
            TensorValue::BF16,
        ),
        DType::F8 => read_table_values::<crate::tensor::F8, _>(
            table,
            selections,
            entry_shape,
            entry_len,
            init,
            output_shape,
            TensorValue::F8,
        ),
        DType::F32 => read_table_values::<f32, _>(table, selections, entry_shape, entry_len, init, output_shape, TensorValue::F32),
        DType::F64 => read_table_values::<f64, _>(table, selections, entry_shape, entry_len, init, output_shape, TensorValue::F64),
        DType::Bool => read_table_values::<bool, _>(table, selections, entry_shape, entry_len, init, output_shape, TensorValue::Bool),
        DType::Bitset => read_table_values::<crate::tensor::Bitset, _>(
            table,
            selections,
            entry_shape,
            entry_len,
            init,
            output_shape,
            TensorValue::Bitset,
        ),
        DType::I4 | DType::I2 | DType::I1 | DType::U4 | DType::U2 | DType::U1 | DType::T2 | DType::T1 => {
            Err(anyhow!("cache table packed dtypes not supported"))
        }
    }
}

fn read_table_values<T: Copy, F>(
    table: &mut CacheTable,
    selections: &[TableIndexSelection],
    entry_shape: &[usize],
    entry_len: usize,
    init: Option<&crate::types::ScalarValue>,
    output_shape: Vec<usize>,
    wrap: F,
) -> Result<TensorValue>
where
    F: Fn(crate::tensor::Tensor<T>) -> TensorValue,
{
    let mut output = Vec::new();
    let mut current = vec![0usize; selections.len()];
    fn recurse<T: Copy>(
        table: &mut CacheTable,
        selections: &[TableIndexSelection],
        entry_shape: &[usize],
        entry_len: usize,
        init: Option<&crate::types::ScalarValue>,
        depth: usize,
        current: &mut [usize],
        output: &mut Vec<T>,
    ) -> Result<()> {
        if depth == selections.len() {
            let entry = get_table_entry::<T>(table, current, entry_shape, entry_len, init)?;
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
    let tensor = crate::tensor::Tensor::from_vec_with_opts(output, crate::tensor::TensorOptions {
        shape: Some(output_shape),
        ..crate::tensor::TensorOptions::default()
    })?;
    Ok(wrap(tensor))
}

fn get_table_entry<T: Copy>(
    table: &mut CacheTable,
    indices: &[usize],
    entry_shape: &[usize],
    entry_len: usize,
    init: Option<&crate::types::ScalarValue>,
) -> Result<Vec<T>> {
    if let Some(entry) = table.entries.get(indices) {
        let entry = match entry {
            TensorValue::I8(t) => unsafe_cast::<i8, T>(&t.data)?,
            TensorValue::I16(t) => unsafe_cast::<i16, T>(&t.data)?,
            TensorValue::I32(t) => unsafe_cast::<i32, T>(&t.data)?,
            TensorValue::I64(t) => unsafe_cast::<i64, T>(&t.data)?,
            TensorValue::U8(t) => unsafe_cast::<u8, T>(&t.data)?,
            TensorValue::U16(t) => unsafe_cast::<u16, T>(&t.data)?,
            TensorValue::U32(t) => unsafe_cast::<u32, T>(&t.data)?,
            TensorValue::U64(t) => unsafe_cast::<u64, T>(&t.data)?,
            TensorValue::F16(t) => unsafe_cast::<crate::tensor::F16, T>(&t.data)?,
            TensorValue::BF16(t) => unsafe_cast::<crate::tensor::BF16, T>(&t.data)?,
            TensorValue::F8(t) => unsafe_cast::<crate::tensor::F8, T>(&t.data)?,
            TensorValue::F32(t) => unsafe_cast::<f32, T>(&t.data)?,
            TensorValue::F64(t) => unsafe_cast::<f64, T>(&t.data)?,
            TensorValue::Bool(t) => unsafe_cast::<bool, T>(&t.data)?,
            TensorValue::Bitset(t) => unsafe_cast::<crate::tensor::Bitset, T>(&t.data)?,
            _ => return Err(anyhow!("cache table entry dtype mismatch")),
        };
        if entry.len() != entry_len {
            return Err(anyhow!("cache table entry length mismatch"));
        }
        return Ok(entry);
    }
    let value = if let Some(init) = init {
        init.to_tensor_value(table.decl.dtype, entry_shape)?
    } else {
        TensorValue::zeros(table.decl.dtype, entry_shape)
    };
    let entry = match &value {
        TensorValue::I8(t) => unsafe_cast::<i8, T>(&t.data)?,
        TensorValue::I16(t) => unsafe_cast::<i16, T>(&t.data)?,
        TensorValue::I32(t) => unsafe_cast::<i32, T>(&t.data)?,
        TensorValue::I64(t) => unsafe_cast::<i64, T>(&t.data)?,
        TensorValue::U8(t) => unsafe_cast::<u8, T>(&t.data)?,
        TensorValue::U16(t) => unsafe_cast::<u16, T>(&t.data)?,
        TensorValue::U32(t) => unsafe_cast::<u32, T>(&t.data)?,
        TensorValue::U64(t) => unsafe_cast::<u64, T>(&t.data)?,
        TensorValue::F16(t) => unsafe_cast::<crate::tensor::F16, T>(&t.data)?,
        TensorValue::BF16(t) => unsafe_cast::<crate::tensor::BF16, T>(&t.data)?,
        TensorValue::F8(t) => unsafe_cast::<crate::tensor::F8, T>(&t.data)?,
        TensorValue::F32(t) => unsafe_cast::<f32, T>(&t.data)?,
        TensorValue::F64(t) => unsafe_cast::<f64, T>(&t.data)?,
        TensorValue::Bool(t) => unsafe_cast::<bool, T>(&t.data)?,
        TensorValue::Bitset(t) => unsafe_cast::<crate::tensor::Bitset, T>(&t.data)?,
        _ => return Err(anyhow!("cache table entry dtype mismatch")),
    };
    if entry.len() != entry_len {
        return Err(anyhow!("cache table entry length mismatch"));
    }
    table.entries.insert(indices.to_vec(), value);
    Ok(entry)
}

fn unsafe_cast<T: Copy, U: Copy>(data: &[T]) -> Result<Vec<U>> {
    let type_mismatch = std::mem::size_of::<T>() != std::mem::size_of::<U>();
    if type_mismatch {
        return Err(anyhow!("cache table entry dtype mismatch"));
    }
    let mut out = Vec::with_capacity(data.len());
    for value in data {
        out.push(unsafe { std::mem::transmute_copy::<T, U>(value) });
    }
    Ok(out)
}
