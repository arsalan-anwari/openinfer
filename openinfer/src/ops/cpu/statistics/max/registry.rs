use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{max_inplace_dispatch, max_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Max, |mode| match mode {
        OpMode::Normal => Some(max_normal_dispatch),
        OpMode::Inplace => Some(max_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build max cpu entries")
});
