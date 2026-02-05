use std::fmt;

use serde::ser::{SerializeStruct, Serializer};
use uuid::Uuid;

/// Kind of trace event emitted during execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum TraceEventKind {
    Assign,
    OpExecute,
    CacheRead,
    CacheWrite,
    CacheIncrement,
    CacheDecrement,
    CacheReset,
    Barrier,
    Dep,
    Transfer,
    Yield,
    Await,
    Branch,
    Loop,
    Return,
}

impl fmt::Display for TraceEventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceEventKind::Assign => write!(f, "Assign"),
            TraceEventKind::OpExecute => write!(f, "OpExecute"),
            TraceEventKind::CacheRead => write!(f, "CacheRead"),
            TraceEventKind::CacheWrite => write!(f, "CacheWrite"),
            TraceEventKind::CacheIncrement => write!(f, "CacheIncrement"),
            TraceEventKind::CacheDecrement => write!(f, "CacheDecrement"),
            TraceEventKind::CacheReset => write!(f, "CacheReset"),
            TraceEventKind::Barrier => write!(f, "Barrier"),
            TraceEventKind::Dep => write!(f, "Dep"),
            TraceEventKind::Transfer => write!(f, "Transfer"),
            TraceEventKind::Yield => write!(f, "Yield"),
            TraceEventKind::Await => write!(f, "Await"),
            TraceEventKind::Branch => write!(f, "Branch"),
            TraceEventKind::Loop => write!(f, "Loop"),
            TraceEventKind::Return => write!(f, "Return"),
        }
    }
}

/// Execution trace record for a single node.
#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub kind: TraceEventKind,
    pub node_index: usize,
    pub node_uuid: Uuid,
    pub block_name: String,
    pub node_desc: String,
    pub op_name: String,
    pub params: Vec<String>,
    pub output: Vec<String>,
    pub micros: String,
    pub micros_parts: [u64; 3],
}

impl serde::Serialize for TraceEvent {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("TraceEvent", 7)?;
        state.serialize_field("block_name", &self.block_name)?;
        state.serialize_field("node_index", &self.node_index)?;
        state.serialize_field("node_uuid", &self.node_uuid)?;
        state.serialize_field("kind", &self.kind)?;
        state.serialize_field("params", &self.params)?;
        state.serialize_field("output", &self.output)?;
        state.serialize_field("micros", &self.micros_parts)?;
        state.end()
    }
}
