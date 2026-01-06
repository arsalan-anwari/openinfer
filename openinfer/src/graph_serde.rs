use anyhow::Result;
use serde_json::Value;

use crate::graph::Graph;

pub struct GraphSerialize;

impl GraphSerialize {
    pub fn json(graph: &Graph) -> Result<Value> {
        Ok(serde_json::to_value(graph)?)
    }
}

pub struct GraphDeserialize;

impl GraphDeserialize {
    pub fn from_json(value: Value) -> Result<Graph> {
        Ok(serde_json::from_value(value)?)
    }
}
