use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{round_inplace_dispatch, round_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Round, |mode| match mode {
        OpMode::Normal => Some(round_normal_dispatch),
        OpMode::Inplace => Some(round_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build round cpu entries")
});
