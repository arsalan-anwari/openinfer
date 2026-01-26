use anyhow::{anyhow, Result};

use super::{ACC_ATTR, ALPHA_ATTR, CLAMP_MAX_ATTR, OpAttrDef, VALUE_ATTR};
use crate::graph::{AttrValue, OpAttrs, OpKind};
use crate::tensor::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
pub enum AccumulateSupport {
    Deny,
    Allow,
}

impl AccumulateSupport {
    pub fn allow(self) -> bool {
        matches!(self, AccumulateSupport::Allow)
    }
}

#[allow(dead_code)]
pub const ACC_INT_PAIRS: &[(DType, DType)] = &[
    (DType::I8, DType::I64),
    (DType::I16, DType::I64),
    (DType::I32, DType::I64),
    (DType::U8, DType::U64),
    (DType::U16, DType::U64),
    (DType::U32, DType::U64),
];

#[allow(dead_code)]
pub const PACKED_ACC_PAIRS: &[(DType, DType)] = &[
    (DType::I1, DType::I64),
    (DType::I2, DType::I64),
    (DType::I4, DType::I64),
    (DType::U1, DType::U64),
    (DType::U2, DType::U64),
    (DType::U4, DType::U64),
];

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct OpSchema {
    pub kind: OpKind,
    pub inputs: usize,
    pub outputs: usize,
    pub attrs: &'static [OpAttrDef],
    pub broadcast: BroadcastSupport,
    pub inplace: InplaceSupport,
    pub accumulate: AccumulateSupport,
    pub type_rule: TypeRule,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum TypeRule {
    SameAsInput(usize),
    Fixed(DType),
    AccFromAttr { attr: &'static str },
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
        inputs: 2,
        outputs: 1,
        attrs: &[ACC_ATTR],
        broadcast: BroadcastSupport::Allow,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Allow,
        type_rule: TypeRule::SameAsInput(0),
    },
    OpSchema {
        kind: OpKind::Mul,
        inputs: 2,
        outputs: 1,
        attrs: &[ACC_ATTR],
        broadcast: BroadcastSupport::Allow,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Allow,
        type_rule: TypeRule::SameAsInput(0),
    },
    OpSchema {
        kind: OpKind::Abs,
        inputs: 1,
        outputs: 1,
        attrs: &[ACC_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Allow,
        type_rule: TypeRule::SameAsInput(0),
    },
    OpSchema {
        kind: OpKind::Relu,
        inputs: 1,
        outputs: 1,
        attrs: &[ALPHA_ATTR, CLAMP_MAX_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
    },
    OpSchema {
        kind: OpKind::Matmul,
        inputs: 2,
        outputs: 1,
        attrs: &[ACC_ATTR],
        broadcast: BroadcastSupport::Allow,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Allow,
        type_rule: TypeRule::SameAsInput(0),
    },
    OpSchema {
        kind: OpKind::IsFinite,
        inputs: 1,
        outputs: 1,
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::Bool),
    },
    OpSchema {
        kind: OpKind::Fill,
        inputs: 1,
        outputs: 1,
        attrs: &[VALUE_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
    },
];

pub fn op_schema(kind: OpKind) -> Option<&'static OpSchema> {
    OPS.iter().find(|op| op.kind == kind)
}
