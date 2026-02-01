use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_with_outputs, KernelFn, OpKey, OpMode};
use crate::op_defs::op_schema;

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    let schema = op_schema(OpKind::Cast).expect("missing cast schema");
    let output_dtypes = schema
        .output_dtypes
        .expect("missing cast output dtypes");
    build_op_entries_with_outputs(OpKind::Cast, output_dtypes, |mode| match mode {
        OpMode::Normal => Some(kernel::cast_normal_dispatch),
        OpMode::Inplace | OpMode::Accumulate => None,
    })
    .expect("failed to build cast vulkan entries")
});
