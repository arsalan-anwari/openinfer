use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{trunc_inplace_dispatch, trunc_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Trunc, |mode| match mode {
        OpMode::Normal => Some(trunc_normal_dispatch),
        OpMode::Inplace => Some(trunc_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build trunc cpu entries")
});
