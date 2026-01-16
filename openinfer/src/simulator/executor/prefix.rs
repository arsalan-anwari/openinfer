use anyhow::{anyhow, Result};

use crate::types::VarDecl;

#[derive(Debug, Clone)]
pub struct PrefixAccess {
    pub base: String,
    pub indices: Vec<String>,
}

pub fn parse_prefix_access(name: &str) -> Result<Option<PrefixAccess>> {
    let open = match name.find('[') {
        Some(pos) => pos,
        None => return Ok(None),
    };
    if !name.ends_with(']') {
        return Err(anyhow!("invalid prefix access {}: missing closing ']'", name));
    }
    let base = name[..open].trim();
    if base.is_empty() {
        return Err(anyhow!("invalid prefix access {}: missing base name", name));
    }
    let inner = &name[open + 1..name.len() - 1];
    if inner.trim().is_empty() {
        return Err(anyhow!("invalid prefix access {}: empty index list", name));
    }
    let mut indices = Vec::new();
    for part in inner.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            return Err(anyhow!("invalid prefix access {}: empty index", name));
        }
        indices.push(trimmed.to_string());
    }
    Ok(Some(PrefixAccess {
        base: base.to_string(),
        indices,
    }))
}

pub fn resolve_prefix_name(decl: &VarDecl, indices: &[String]) -> Result<String> {
    if indices.len() != decl.table_indices.len() {
        return Err(anyhow!(
            "prefix access for {} expects {} indices, got {}",
            decl.name,
            decl.table_indices.len(),
            indices.len()
        ));
    }
    let mut resolved = decl
        .pattern
        .as_ref()
        .ok_or_else(|| anyhow!("prefix table {} is missing @pattern", decl.name))?
        .clone();
    for (name, value) in decl.table_indices.iter().zip(indices.iter()) {
        let placeholder = format!("{{{}}}", name);
        resolved = resolved.replace(&placeholder, value);
    }
    Ok(resolved)
}
