use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{neg_inplace_dispatch, neg_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Neg, |mode| match mode {
        OpMode::Normal => Some(neg_normal_dispatch),
        OpMode::Inplace => Some(neg_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build neg cpu entries")
});
