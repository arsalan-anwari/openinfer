#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum OpAttrType {
    Scalar,
    DType,
    Tensor,
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
