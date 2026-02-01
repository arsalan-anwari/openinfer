use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{sub_accumulate_dispatch, sub_inplace_dispatch, sub_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Sub, |mode| match mode {
        OpMode::Normal => Some(sub_normal_dispatch),
        OpMode::Inplace => Some(sub_inplace_dispatch),
        OpMode::Accumulate => Some(sub_accumulate_dispatch),
    })
    .expect("failed to build sub cpu entries")
});
