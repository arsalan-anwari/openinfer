#![allow(unused_imports)]

mod op_attrs;
mod op_defs;
mod var_attrs;

pub use op_attrs::{OpAttrDef, OpAttrType, ACC_ATTR, ALPHA_ATTR, CLAMP_MAX_ATTR, VALUE_ATTR};
pub use op_defs::{op_def, OpDef, OPS};
pub use var_attrs::{VarAttrDef, VAR_ATTRS};
