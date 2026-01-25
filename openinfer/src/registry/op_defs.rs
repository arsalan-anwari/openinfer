use super::{ACC_ATTR, ALPHA_ATTR, CLAMP_MAX_ATTR, OpAttrDef, VALUE_ATTR};

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct OpDef {
    pub name: &'static str,
    pub inputs: usize,
    pub outputs: usize,
    pub attrs: &'static [OpAttrDef],
    pub supports_broadcast: bool,
    pub supports_inplace: bool,
    pub supports_accumulate: bool,
}

pub const OPS: &[OpDef] = &[
    OpDef {
        name: "add",
        inputs: 2,
        outputs: 1,
        attrs: &[ACC_ATTR],
        supports_broadcast: true,
        supports_inplace: true,
        supports_accumulate: true,
    },
    OpDef {
        name: "mul",
        inputs: 2,
        outputs: 1,
        attrs: &[ACC_ATTR],
        supports_broadcast: true,
        supports_inplace: true,
        supports_accumulate: true,
    },
    OpDef {
        name: "abs",
        inputs: 1,
        outputs: 1,
        attrs: &[ACC_ATTR],
        supports_broadcast: false,
        supports_inplace: true,
        supports_accumulate: true,
    },
    OpDef {
        name: "relu",
        inputs: 1,
        outputs: 1,
        attrs: &[ALPHA_ATTR, CLAMP_MAX_ATTR],
        supports_broadcast: false,
        supports_inplace: true,
        supports_accumulate: false,
    },
    OpDef {
        name: "matmul",
        inputs: 2,
        outputs: 1,
        attrs: &[ACC_ATTR],
        supports_broadcast: true,
        supports_inplace: true,
        supports_accumulate: true,
    },
    OpDef {
        name: "is_finite",
        inputs: 1,
        outputs: 1,
        attrs: &[],
        supports_broadcast: false,
        supports_inplace: false,
        supports_accumulate: false,
    },
    OpDef {
        name: "fill",
        inputs: 1,
        outputs: 1,
        attrs: &[VALUE_ATTR],
        supports_broadcast: false,
        supports_inplace: true,
        supports_accumulate: false,
    },
];

pub fn op_def(name: &str) -> Option<&'static OpDef> {
    OPS.iter().find(|op| op.name == name)
}
