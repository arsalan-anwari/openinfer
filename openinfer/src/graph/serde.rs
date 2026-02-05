use anyhow::Result;
use serde_json::Value;

use crate::graph::Graph;

/// Serialize graphs into JSON representations.
pub struct GraphSerialize;

impl GraphSerialize {
    /// Convert a graph to a JSON value.
    pub fn json(graph: &Graph) -> Result<Value> {
        Ok(serde_json::to_value(graph)?)
    }
}

/// Deserialize graphs from JSON representations.
pub struct GraphDeserialize;

impl GraphDeserialize {
    /// Parse a graph from a JSON value.
    pub fn from_json(value: Value) -> Result<Graph> {
        Ok(serde_json::from_value(value)?)
    }
}
