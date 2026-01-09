use anyhow::{Context, Result};

use super::ValidationContext;

pub(crate) fn validate_dims(ctx: &ValidationContext, dims: &[String], name: &str) -> Result<()> {
    for dim in dims {
        if dim.parse::<usize>().is_ok() {
            continue;
        }
        ctx.model
            .size_of(dim)
            .with_context(|| format!("unknown sizevar {} for {}", dim, name))?;
    }
    Ok(())
}
