use std::collections::HashMap;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::tensor::DType;
use crate::types::ScalarValue;

use super::var::{MemoryKind, VarDecl};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttrValue {
    Float(f32),
    Double(f64),
    Int(i64),
    UInt(u64),
    Bool(bool),
    Var(String),
    DType(DType),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CacheIndexValue {
    Ident(String),
    Lit(i64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CacheIndexExpr {
    Single(CacheIndexValue),
    Slice {
        start: Option<CacheIndexValue>,
        end: Option<CacheIndexValue>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CacheAccess {
    pub base: String,
    pub indices: Vec<CacheIndexExpr>,
    pub bracketed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpAttr {
    pub name: String,
    pub value: AttrValue,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpAttrs {
    pub items: Vec<OpAttr>,
}

impl OpAttrs {
    pub fn none() -> Self {
        Self { items: Vec::new() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    Assign { name: String, dtype: DType, dims: Vec<String> },
    Op {
        op: String,
        attrs: OpAttrs,
        inputs: Vec<String>,
        output: String,
    },
    Branch {
        cond: Option<String>,
        then_block: String,
        else_block: Option<String>,
    },
    CacheRead {
        src: CacheAccess,
        dst: String,
    },
    CacheWrite {
        src: String,
        dst: CacheAccess,
    },
    CacheIncrement {
        target: String,
        amount: i64,
    },
    CacheDecrement {
        target: String,
        amount: i64,
    },
    CacheReset {
        target: CacheAccess,
    },
    Yield {
        vars: Vec<String>,
    },
    Await {
        vars: Vec<String>,
    },
    Loop {
        name: String,
        index: String,
        start: String,
        end: String,
        body: Vec<Node>,
    },
    Return,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub index: usize,
    pub uuid: Uuid,
    pub kind: NodeKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub name: String,
    pub nodes: Vec<Node>,
}

impl Block {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub vars: HashMap<String, VarDecl>,
    pub blocks: HashMap<String, Block>,
    next_index: usize,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            blocks: HashMap::new(),
            next_index: 0,
        }
    }

    pub fn add_var(
        &mut self,
        kind: MemoryKind,
        name: impl Into<String>,
        dtype: DType,
        dims: Vec<String>,
        init: Option<ScalarValue>,
        ref_name: Option<String>,
        table_indices: Vec<String>,
        pattern: Option<String>,
        table: bool,
        auto_dim: Vec<String>,
        fixed: Vec<(String, usize)>,
    ) {
        let name = name.into();
        self.vars.insert(
            name.clone(),
            VarDecl {
                name,
                ref_name,
                pattern,
                table_indices,
                table,
                auto_dim,
                fixed,
                dtype,
                dims,
                kind,
                init,
            },
        );
    }

    pub fn add_block(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.blocks.entry(name.clone()).or_insert_with(|| Block::new(name));
    }

    pub fn add_node(&mut self, block: &str, kind: NodeKind) -> Result<()> {
        let node = self.make_node(kind);
        let block = self
            .blocks
            .get_mut(block)
            .ok_or_else(|| anyhow!("missing block: {}", block))?;
        block.nodes.push(node);
        Ok(())
    }

    pub fn add_prebuilt_node(&mut self, block: &str, node: Node) -> Result<()> {
        let block = self
            .blocks
            .get_mut(block)
            .ok_or_else(|| anyhow!("missing block: {}", block))?;
        block.nodes.push(node);
        Ok(())
    }

    pub fn make_node(&mut self, kind: NodeKind) -> Node {
        let node = Node {
            index: self.next_index,
            uuid: Uuid::new_v4(),
            kind,
        };
        self.next_index += 1;
        node
    }

    pub fn make_loop_node(
        &mut self,
        name: impl Into<String>,
        index: impl Into<String>,
        start: impl Into<String>,
        end: impl Into<String>,
        body: Vec<Node>,
    ) -> Node {
        let loop_index = self.next_index;
        self.next_index += 1;
        let kind = NodeKind::Loop {
            name: name.into(),
            index: index.into(),
            start: start.into(),
            end: end.into(),
            body,
        };
        Node {
            index: loop_index,
            uuid: Uuid::new_v4(),
            kind,
        }
    }

    pub fn block(&self, name: &str) -> Result<&Block> {
        self.blocks
            .get(name)
            .ok_or_else(|| anyhow!("missing block: {}", name))
    }
}
