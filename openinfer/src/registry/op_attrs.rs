#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum OpAttrType {
    Scalar,
    DType,
    Tensor,
    String,
    IntList,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ScalarAttrKind {
    Float,
    Int,
    UInt,
    Bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpAttrDef {
    pub name: &'static str,
    pub kind: OpAttrType,
    pub scalar_kinds: &'static [ScalarAttrKind],
}

impl OpAttrDef {
    pub const fn new(name: &'static str, kind: OpAttrType) -> Self {
        Self {
            name,
            kind,
            scalar_kinds: &[],
        }
    }

    pub const fn scalar(name: &'static str, scalar_kinds: &'static [ScalarAttrKind]) -> Self {
        Self {
            name,
            kind: OpAttrType::Scalar,
            scalar_kinds,
        }
    }
}

pub const ACC_ATTR: OpAttrDef = OpAttrDef::new("acc", OpAttrType::DType);

pub const NUMERIC_SCALAR_KINDS: &[ScalarAttrKind] = &[
    ScalarAttrKind::Float,
    ScalarAttrKind::Int,
    ScalarAttrKind::UInt,
    ScalarAttrKind::Bool,
];

#[allow(dead_code)]
pub const ALPHA_ATTR: OpAttrDef = OpAttrDef::scalar("alpha", NUMERIC_SCALAR_KINDS);
#[allow(dead_code)]
pub const CLAMP_MAX_ATTR: OpAttrDef = OpAttrDef::scalar("clamp_max", NUMERIC_SCALAR_KINDS);
#[allow(dead_code)]
pub const VALUE_ATTR: OpAttrDef = OpAttrDef::scalar("value", NUMERIC_SCALAR_KINDS);

#[allow(dead_code)]
pub const DIV_BY_ZERO_MASK_ATTR: OpAttrDef =
    OpAttrDef::scalar("div_by_zero_mask", NUMERIC_SCALAR_KINDS);
#[allow(dead_code)]
pub const MIN_ATTR: OpAttrDef = OpAttrDef::scalar("min", NUMERIC_SCALAR_KINDS);
#[allow(dead_code)]
pub const MAX_ATTR: OpAttrDef = OpAttrDef::scalar("max", NUMERIC_SCALAR_KINDS);
#[allow(dead_code)]
pub const BITS_ATTR: OpAttrDef = OpAttrDef::scalar("bits", NUMERIC_SCALAR_KINDS);
#[allow(dead_code)]
pub const KEEPDIMS_ATTR: OpAttrDef = OpAttrDef::scalar("keepdims", &[ScalarAttrKind::Bool]);
#[allow(dead_code)]
pub const AXIS_ATTR: OpAttrDef = OpAttrDef::scalar("axis", NUMERIC_SCALAR_KINDS);
#[allow(dead_code)]
pub const SELECT_FIRST_ATTR: OpAttrDef =
    OpAttrDef::scalar("select_first", &[ScalarAttrKind::Bool]);
#[allow(dead_code)]
pub const SATURATE_ATTR: OpAttrDef = OpAttrDef::scalar("saturate", &[ScalarAttrKind::Bool]);

#[allow(dead_code)]
pub const AXES_ATTR: OpAttrDef = OpAttrDef::new("axes", OpAttrType::IntList);
#[allow(dead_code)]
pub const ROUNDING_MODE_ATTR: OpAttrDef = OpAttrDef::new("rounding_mode", OpAttrType::String);
#[allow(dead_code)]
pub const TO_ATTR: OpAttrDef = OpAttrDef::new("to", OpAttrType::DType);
