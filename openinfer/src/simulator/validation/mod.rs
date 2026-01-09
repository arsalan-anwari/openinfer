use anyhow::Result;

use crate::graph::Graph;
use crate::model_loader::ModelLoader;

use super::Device;

mod attrs;
mod block;
mod dims;
mod graph;
mod vars;

pub(crate) struct ValidationContext<'a> {
    pub model: &'a ModelLoader,
    pub graph: &'a Graph,
    pub device: Device,
}

impl<'a> ValidationContext<'a> {
    pub(crate) fn new(model: &'a ModelLoader, graph: &'a Graph, device: Device) -> Self {
        Self {
            model,
            graph,
            device,
        }
    }
}

pub(crate) fn validate_graph(model: &ModelLoader, graph: &Graph, device: Device) -> Result<()> {
    let ctx = ValidationContext::new(model, graph, device);
    graph::validate_graph(&ctx)
}
