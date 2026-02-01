use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{ceil_inplace_dispatch, ceil_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Ceil, |mode| match mode {
        OpMode::Normal => Some(ceil_normal_dispatch),
        OpMode::Inplace => Some(ceil_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build ceil cpu entries")
});
