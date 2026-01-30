use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{and_inplace_dispatch, and_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::And, |mode| match mode {
        OpMode::Normal => Some(and_normal_dispatch),
        OpMode::Inplace => Some(and_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build and cpu entries")
});
