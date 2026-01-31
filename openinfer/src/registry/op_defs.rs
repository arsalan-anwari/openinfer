use anyhow::{anyhow, Result};

use super::{
    ACC_ATTR, ALPHA_ATTR, AXES_ATTR, AXIS_ATTR, BITS_ATTR, CLAMP_MAX_ATTR,
    DIV_BY_ZERO_MASK_ATTR, KEEPDIMS_ATTR, MAX_ATTR, MIN_ATTR, OpAttrDef, OpDTypeSupport,
    ROUNDING_MODE_ATTR, SATURATE_ATTR, SELECT_FIRST_ATTR, TO_ATTR, VALUE_ATTR,
};
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
    OpSchema {
        kind: OpKind::Sub,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[ACC_ATTR],
        broadcast: BroadcastSupport::Allow,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Allow,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::SUB_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Div,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[DIV_BY_ZERO_MASK_ATTR],
        broadcast: BroadcastSupport::Allow,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::DIV_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::FloorDiv,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[DIV_BY_ZERO_MASK_ATTR],
        broadcast: BroadcastSupport::Allow,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::FLOOR_DIV_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Rem,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Allow,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::REM_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Fma,
        inputs: InputArity::Fixed(3),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::FMA_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Neg,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::NEG_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Sign,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::I8),
        dtype_support: Some(&super::SIGN_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Recip,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[DIV_BY_ZERO_MASK_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::RECIP_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Min,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::MIN_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Max,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::MAX_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Clamp,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[MIN_ATTR, MAX_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::CLAMP_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Floor,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::FLOOR_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Ceil,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::CEIL_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Round,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::ROUND_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Trunc,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::TRUNC_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::And,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::AND_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Or,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::OR_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Xor,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::XOR_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Not,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::NOT_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Shl,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[BITS_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::SHL_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Shr,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[BITS_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Allow,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::SHR_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Popcount,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::U8),
        dtype_support: Some(&super::POPCOUNT_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Eq,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::Bool),
        dtype_support: Some(&super::EQ_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Ne,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::Bool),
        dtype_support: Some(&super::NE_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Lt,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::Bool),
        dtype_support: Some(&super::LT_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Le,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::Bool),
        dtype_support: Some(&super::LE_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Gt,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::Bool),
        dtype_support: Some(&super::GT_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Ge,
        inputs: InputArity::Fixed(2),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::Bool),
        dtype_support: Some(&super::GE_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Filter,
        inputs: InputArity::Fixed(3),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::FILTER_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::IsNan,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::Bool),
        dtype_support: Some(&super::IS_NAN_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::IsInf,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::Bool),
        dtype_support: Some(&super::IS_INF_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::IsNeg,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::Bool),
        dtype_support: Some(&super::IS_NEG_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::SumAxis,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[AXES_ATTR, KEEPDIMS_ATTR, ACC_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Allow,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::SUM_AXIS_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::MeanAxis,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[AXES_ATTR, KEEPDIMS_ATTR, ACC_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Allow,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::MEAN_AXIS_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::ProdAxis,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[AXES_ATTR, KEEPDIMS_ATTR, ACC_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Allow,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::PROD_AXIS_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::MaxAxis,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[AXES_ATTR, KEEPDIMS_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::MAX_AXIS_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::MinAxis,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[AXES_ATTR, KEEPDIMS_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::SameAsInput(0),
        dtype_support: Some(&super::MIN_AXIS_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::ArgmaxAxis,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[AXIS_ATTR, KEEPDIMS_ATTR, SELECT_FIRST_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::I64),
        dtype_support: Some(&super::ARGMAX_AXIS_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::ArgminAxis,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[AXIS_ATTR, KEEPDIMS_ATTR, SELECT_FIRST_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::Fixed(DType::I64),
        dtype_support: Some(&super::ARGMIN_AXIS_DTYPE_SUPPORT),
    },
    OpSchema {
        kind: OpKind::Cast,
        inputs: InputArity::Fixed(1),
        outputs: OutputArity::Fixed(1),
        attrs: &[TO_ATTR, ROUNDING_MODE_ATTR, SATURATE_ATTR],
        broadcast: BroadcastSupport::Deny,
        inplace: InplaceSupport::Deny,
        accumulate: AccumulateSupport::Deny,
        type_rule: TypeRule::AccFromAttr { attr: "to" },
        dtype_support: Some(&super::CAST_DTYPE_SUPPORT),
    }
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
