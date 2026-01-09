use std::collections::HashSet;

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
    validate_prefix_decl(decl)?;
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

pub fn validate_prefix_decl(decl: &VarDecl) -> Result<()> {
    if decl.table_indices.is_empty() && decl.pattern.is_none() {
        return Ok(());
    }
    if decl.table_indices.is_empty() {
        return Err(anyhow!(
            "prefix table {} must declare at least one index",
            decl.name
        ));
    }
    if decl.pattern.is_none() {
        return Err(anyhow!("prefix table {} is missing @pattern", decl.name));
    }
    let mut seen = HashSet::new();
    for index in &decl.table_indices {
        if !seen.insert(index) {
            return Err(anyhow!(
                "prefix table {} has duplicate index {}",
                decl.name,
                index
            ));
        }
    }
    let placeholders = parse_placeholders(
        decl.pattern
            .as_ref()
            .ok_or_else(|| anyhow!("prefix table {} is missing @pattern", decl.name))?,
    )?;
    let placeholder_set: HashSet<String> = placeholders.iter().cloned().collect();
    for index in &decl.table_indices {
        if !placeholder_set.contains(index) {
            return Err(anyhow!(
                "pattern for {} is missing placeholder {{{}}}",
                decl.name,
                index
            ));
        }
    }
    for placeholder in placeholder_set {
        if !decl.table_indices.contains(&placeholder) {
            return Err(anyhow!(
                "pattern for {} references unknown index {{{}}}",
                decl.name,
                placeholder
            ));
        }
    }
    Ok(())
}

fn parse_placeholders(pattern: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut chars = pattern.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '{' {
            continue;
        }
        let mut name = String::new();
        let mut closed = false;
        while let Some(next) = chars.next() {
            if next == '}' {
                closed = true;
                break;
            }
            name.push(next);
        }
        if !closed {
            return Err(anyhow!("pattern placeholder missing closing brace"));
        }
        if name.is_empty() {
            return Err(anyhow!("pattern placeholder cannot be empty"));
        }
        if !is_ident(&name) {
            return Err(anyhow!("invalid pattern placeholder {{{}}}", name));
        }
        out.push(name);
    }
    Ok(out)
}

fn is_ident(value: &str) -> bool {
    let mut chars = value.chars();
    match chars.next() {
        Some(ch) if ch == '_' || ch.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}
