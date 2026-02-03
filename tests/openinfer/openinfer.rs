#[path = "common/mod.rs"]
mod common;

#[path = "ops/ops_basic.rs"]
mod ops_basic;
#[path = "ops/ops_matmul.rs"]
mod ops_matmul;
#[path = "ops/ops_broadcast.rs"]
mod ops_broadcast;
#[path = "ops/ops_compare.rs"]
mod ops_compare;
#[path = "ops/ops_bitwise.rs"]
mod ops_bitwise;
#[path = "ops/ops_rounding.rs"]
mod ops_rounding;
#[path = "ops/ops_reduce.rs"]
mod ops_reduce;
#[path = "ops/ops_cast.rs"]
mod ops_cast;
#[path = "ops/ops_accumulate_inplace.rs"]
mod ops_accumulate_inplace;
#[path = "ops/ops_misc.rs"]
mod ops_misc;
#[path = "ops/ops_float_special.rs"]
mod ops_float_special;
#[path = "ops/ops_packed.rs"]
mod ops_packed;
#[path = "ops/ops_full_matrix.rs"]
mod ops_full_matrix;

#[path = "graph/graph_simple.rs"]
mod graph_simple;
#[path = "graph/graph_branch.rs"]
mod graph_branch;
#[path = "graph/graph_examples.rs"]
mod graph_examples;
#[path = "graph/graph_more.rs"]
mod graph_more;
#[path = "graph/graph_cache.rs"]
mod graph_cache;
#[path = "graph/graph_control_flow.rs"]
mod graph_control_flow;
#[path = "graph/graph_serde.rs"]
mod graph_serde;

#[path = "tensor/tensor_layout.rs"]
mod tensor_layout;
#[path = "tensor/tensor_dtypes.rs"]
mod tensor_dtypes;
