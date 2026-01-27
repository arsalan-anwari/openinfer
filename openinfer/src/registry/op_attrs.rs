#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum OpAttrType {
    Scalar,
    DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpAttrDef {
    pub name: &'static str,
    pub kind: OpAttrType,
}

impl OpAttrDef {
    pub const fn new(name: &'static str, kind: OpAttrType) -> Self {
        Self { name, kind }
    }
}

pub const ACC_ATTR: OpAttrDef = OpAttrDef::new("acc", OpAttrType::DType);
#[allow(dead_code)]
pub const ALPHA_ATTR: OpAttrDef = OpAttrDef::new("alpha", OpAttrType::Scalar);
#[allow(dead_code)]
pub const CLAMP_MAX_ATTR: OpAttrDef = OpAttrDef::new("clamp_max", OpAttrType::Scalar);
#[allow(dead_code)]
pub const VALUE_ATTR: OpAttrDef = OpAttrDef::new("value", OpAttrType::Scalar);
