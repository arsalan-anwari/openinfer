use anyhow::{anyhow, Result};

use crate::graph::VarDecl;

#[derive(Debug, Clone)]
pub struct AutoDimState {
    pub base_shape: Vec<usize>,
    pub counts: Vec<usize>,
    pub max: Vec<Option<usize>>,
}

pub fn auto_dim_shape(base: &[usize], counts: &[usize]) -> Result<Vec<usize>> {
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

pub fn fixed_size_for(decl: &VarDecl, index: &str) -> Option<usize> {
    decl.fixed
        .iter()
        .find(|(name, _)| name == index)
        .map(|(_, value)| *value)
}

pub fn init_cache_table_sizes(
    decl: &VarDecl,
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
