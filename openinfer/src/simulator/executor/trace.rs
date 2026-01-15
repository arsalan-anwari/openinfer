use std::fmt;

use anyhow::Result;
use serde::ser::{SerializeStruct, Serializer};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum TraceEventKind {
    Assign,
    OpExecute,
    CacheRead,
    CacheWrite,
    CacheIncrement,
    CacheDecrement,
    CacheReset,
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
            TraceEventKind::Return => write!(f, "Return"),
        }
    }
}

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

pub(crate) fn format_duration_ns(ns: u128) -> (String, [u64; 3]) {
    let ms = (ns / 1_000_000) as u64;
    let rem_ms = (ns % 1_000_000) as u64;
    let us = rem_ms / 1_000;
    let ns = rem_ms % 1_000;
    (format!("{}ms {}us {}ns", ms, us, ns), [ms, us, ns])
}

pub(crate) fn format_step_line(event: &TraceEvent) -> String {
    match event.kind {
        TraceEventKind::OpExecute => format!(
            "{} {} [{}] -- {} -- ({})",
            event.node_index, event.node_uuid, event.block_name, event.node_desc, event.micros
        ),
        _ => format!(
            "{} {} [{}] -- {}",
            event.node_index, event.node_uuid, event.block_name, event.node_desc
        ),
    }
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
