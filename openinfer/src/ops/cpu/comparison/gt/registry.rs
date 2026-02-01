use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::gt_normal_dispatch;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Gt, |mode| match mode {
        OpMode::Normal => Some(gt_normal_dispatch),
        OpMode::Inplace => None,
        OpMode::Accumulate => None,
    })
    .expect("failed to build gt cpu entries")
});
