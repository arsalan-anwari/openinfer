#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub struct VarAttrDef {
    pub name: &'static str,
    pub description: &'static str,
}

#[allow(dead_code)]
pub const VAR_ATTRS: &[VarAttrDef] = &[
    VarAttrDef {
        name: "init",
        description: "Initialize variable with a scalar literal.",
    },
    VarAttrDef {
        name: "ref",
        description: "Bind variable to a model tensor by name.",
    },
    VarAttrDef {
        name: "pattern",
        description: "Prefix-table pattern for matching cache variables.",
    },
    VarAttrDef {
        name: "table",
        description: "Mark persistent variable as a cache table.",
    },
    VarAttrDef {
        name: "auto_dim",
        description: "Auto-index dimensions for cache tables.",
    },
    VarAttrDef {
        name: "fixed",
        description: "Fix a dimension to a specific size for cache tables.",
    },
];
