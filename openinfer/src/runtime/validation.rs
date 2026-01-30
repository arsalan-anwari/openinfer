mod attrs;
mod blocks;
mod cache;
mod context;
mod control_flow;
mod ops;
mod vars;

use anyhow::Result;

use crate::graph::Graph;
use crate::runtime::ModelLoader;

pub fn validate_graph(model: &ModelLoader, graph: &Graph) -> Result<()> {
    let mut ctx = context::ValidationContext::new(model, graph);
    vars::validate_vars(&mut ctx)?;
    blocks::validate_blocks(&mut ctx)?;
    Ok(())
}
