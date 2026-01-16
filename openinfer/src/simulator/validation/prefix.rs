use std::collections::HashSet;

use anyhow::{anyhow, Result};

use crate::types::VarDecl;

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
