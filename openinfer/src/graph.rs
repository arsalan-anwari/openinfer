use std::collections::HashMap;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::tensor::DType;
use crate::types::{MemoryKind, ScalarValue, VarDecl};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeKind {
    Assign { name: String, dtype: DType, dims: Vec<String> },
    Op {
        name: String,
        inputs: Vec<String>,
        output: String,
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
    ) {
        let name = name.into();
        self.vars.insert(
            name.clone(),
            VarDecl {
                name,
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
        let block = self
            .blocks
            .get_mut(block)
            .ok_or_else(|| anyhow!("missing block: {}", block))?;
        let node = Node {
            index: self.next_index,
            uuid: Uuid::new_v4(),
            kind,
        };
        self.next_index += 1;
        block.nodes.push(node);
        Ok(())
    }

    pub fn block(&self, name: &str) -> Result<&Block> {
        self.blocks
            .get(name)
            .ok_or_else(|| anyhow!("missing block: {}", name))
    }
}

pub fn describe_node(kind: &NodeKind) -> String {
    match kind {
        NodeKind::Assign { name, .. } => format!("assign {}", name),
        NodeKind::Op { name, inputs, output } => {
            format!("op {}({}) >> {}", name, inputs.join(","), output)
        }
        NodeKind::Return => "return".to_string(),
    }
}
