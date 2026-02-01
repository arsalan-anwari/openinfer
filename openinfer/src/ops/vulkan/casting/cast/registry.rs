use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_with_outputs, KernelFn, OpKey, OpMode};
use crate::registry::CAST_OUTPUT_DTYPES;

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_with_outputs(OpKind::Cast, CAST_OUTPUT_DTYPES, |mode| match mode {
        OpMode::Normal => Some(kernel::cast_normal_dispatch),
        OpMode::Inplace | OpMode::Accumulate => None,
    })
    .expect("failed to build cast vulkan entries")
});
