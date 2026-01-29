use anyhow::{anyhow, Result};

use super::{ACC_ATTR, ALPHA_ATTR, CLAMP_MAX_ATTR, OpAttrDef, OpDTypeSupport, VALUE_ATTR};
use crate::graph::{AttrValue, OpAttrs, OpKind};
use crate::tensor::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BroadcastSupport {
    Deny,
    Allow,
}

impl BroadcastSupport {
    pub fn allow(self) -> bool {
        matches!(self, BroadcastSupport::Allow)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum InplaceSupport {
    Deny,
    Allow,
}

impl InplaceSupport {
    pub fn allow(self) -> bool {
        matches!(self, InplaceSupport::Allow)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AccumulateSupport {
    Deny,
    Allow,
}

impl AccumulateSupport {
    pub fn allow(self) -> bool {
        matches!(self, AccumulateSupport::Allow)
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct OpSchema {
    pub kind: OpKind,
    pub inputs: InputArity,
    pub outputs: OutputArity,
    pub attrs: &'static [OpAttrDef],
    pub broadcast: BroadcastSupport,
    pub inplace: InplaceSupport,
    pub accumulate: AccumulateSupport,
    pub type_rule: TypeRule,
    pub dtype_support: Option<&'static OpDTypeSupport>,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum TypeRule {
    SameAsInput(usize),
    Fixed(DType),
    AccFromAttr { attr: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum InputArity {
    Fixed(usize),
    AtLeast(usize),
    Any,
}

impl InputArity {
    pub fn allows(self, count: usize) -> bool {
        match self {
            InputArity::Fixed(expected) => count == expected,
            InputArity::AtLeast(min) => count >= min,
            InputArity::Any => true,
        }
    }

    pub fn fixed(self) -> Option<usize> {
        match self {
            InputArity::Fixed(count) => Some(count),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum OutputArity {
    Fixed(usize),
    AtLeast(usize),
    Any,
}

#[allow(dead_code)]
impl OutputArity {
    pub fn allows(self, count: usize) -> bool {
        match self {
            OutputArity::Fixed(expected) => count == expected,
            OutputArity::AtLeast(min) => count >= min,
            OutputArity::Any => true,
        }
    }

    #[allow(dead_code)]
    pub fn fixed(self) -> Option<usize> {
        match self {
            OutputArity::Fixed(count) => Some(count),
            _ => None,
        }
    }
}

impl TypeRule {
    pub fn output_dtype(self, inputs: &[DType], attrs: &OpAttrs) -> Result<DType> {
        match self {
            TypeRule::SameAsInput(index) => inputs
                .get(index)
                .copied()
                .ok_or_else(|| anyhow!("missing input dtype at {}", index)),
            TypeRule::Fixed(dtype) => Ok(dtype),
            TypeRule::AccFromAttr { attr } => attrs
                .items
                .iter()
                .find(|item| item.name == attr)
                .ok_or_else(|| anyhow!("missing {} attribute", attr))
                .and_then(|item| match &item.value {
                    AttrValue::DType(dtype) => Ok(*dtype),
                    _ => Err(anyhow!("{} attribute must be a dtype", attr)),
                }),
        }
    }
}

pub const OPS: &[OpSchema] = &[
    OpSchema {
        kind: OpKind::Add,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[ACC_ATTR],
        broadcast: BroadcastSupport::Allow,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Allow,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::ADD_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Mul,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[ACC_ATTR],
        broadcast: BroadcastSupport::Allow,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Allow,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::MUL_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Abs,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[ACC_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Allow,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::ABS_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Relu,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[ALPHA_ATTR, CLAMP_MAX_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::RELU_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Matmul,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[ACC_ATTR],
        broadcast: BroadcastSupport::Allow,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Allow,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::MATMUL_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::IsFinite,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::Bool),
        dtype_support: Some(&super::IS_FINITE_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Fill,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[VALUE_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::FILL_DTYPE_SUPPORT),
    },
];

#[allow(unused)]
pub fn acc_dtype(attrs: &OpAttrs) -> Result<DType> {
    attrs
        .items
        .iter()
        .find(|attr| attr.name == "acc")
        .ok_or_else(|| anyhow!("missing acc attribute"))
        .and_then(|attr| match &attr.value {
            AttrValue::DType(dtype) => Ok(*dtype),
            _ => Err(anyhow!("acc attribute must be a dtype")),
        })
}

pub fn op_schema(kind: OpKind) -> Option<&'static OpSchema> {
    OPS.iter().find(|op| op.kind == kind)
}
