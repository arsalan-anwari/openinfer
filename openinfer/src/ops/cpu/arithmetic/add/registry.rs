use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{add_accumulate_dispatch, add_inplace_dispatch, add_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Add, |mode| match mode {
        OpMode::Normal => Some(add_normal_dispatch),
        OpMode::Inplace => Some(add_inplace_dispatch),
        OpMode::Accumulate => Some(add_accumulate_dispatch),
    })
    .expect("failed to build add cpu entries")
});
