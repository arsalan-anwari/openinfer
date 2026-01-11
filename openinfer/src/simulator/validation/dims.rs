use anyhow::{Context, Result};

use super::ValidationContext;

pub(crate) fn validate_dims(ctx: &ValidationContext, dims: &[String], name: &str) -> Result<()> {
    for dim in dims {
        let trimmed = dim.trim();
        if trimmed.parse::<usize>().is_ok() {
            continue;
        }
        if let Some((left, right)) = trimmed.split_once('*') {
            for part in [left.trim(), right.trim()] {
                if part.parse::<usize>().is_ok() {
                    continue;
                }
                ctx.model
                    .size_of(part)
                    .with_context(|| format!("unknown sizevar {} for {}", part, name))?;
            }
            continue;
        }
        ctx.model
            .size_of(trimmed)
            .with_context(|| format!("unknown sizevar {} for {}", trimmed, name))?;
    }
    Ok(())
}
