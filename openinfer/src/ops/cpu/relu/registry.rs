use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{relu_inplace_dispatch, relu_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Relu, |mode| match mode {
        OpMode::Normal => Some(relu_normal_dispatch),
        OpMode::Inplace => Some(relu_inplace_dispatch),
        OpMode::Accumulate => None,
    })
    .expect("failed to build relu cpu entries")
});
