//! Graph data structures and serialization helpers.
//!
//! The `graph` module defines the core graph model (`Graph`, `Block`, `Node`)
//! along with operation kinds and attribute types used by the runtime.
//!
//! ## Highlights
//! - `NodeKind` variants encode control flow, ops, and cache operations.
//! - `GraphSerialize` / `GraphDeserialize` support JSON conversion.
mod node;
mod serde;
mod types;
mod var;

pub use node::describe_node;
pub use serde::{GraphDeserialize, GraphSerialize};
pub use types::{
    AttrValue, Block, CacheAccess, CacheIndexExpr, CacheIndexValue, Graph, Node, NodeKind, OpAttr,
    OpAttrs, OpKind,
};
pub use var::{MemoryKind, VarDecl};
