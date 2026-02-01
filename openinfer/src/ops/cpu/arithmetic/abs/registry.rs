use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{abs_accumulate_dispatch, abs_inplace_dispatch, abs_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Abs, |mode| match mode {
        OpMode::Normal => Some(abs_normal_dispatch),
        OpMode::Inplace => Some(abs_inplace_dispatch),
        OpMode::Accumulate => Some(abs_accumulate_dispatch),
    })
    .expect("failed to build abs cpu entries")
});
