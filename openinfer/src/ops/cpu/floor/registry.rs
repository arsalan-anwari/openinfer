use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{floor_inplace_dispatch, floor_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Floor, |mode| match mode {
        OpMode::Normal => Some(floor_normal_dispatch),
        OpMode::Inplace => Some(floor_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build floor cpu entries")
});
