mod auto_dim;
mod store;
mod table;
mod utils;

pub use auto_dim::{auto_dim_shape, fixed_size_for, init_cache_table_sizes, AutoDimState};
pub use store::CacheStore;
pub use table::{
    read_table_selection, recompute_cache_table_sizes, reset_cache_table_sizes,
    resolve_table_indices_from_resolved, resolve_table_prefix_indices_from_resolved, CacheTable,
};
pub use utils::{expand_tensor_value, increment_scalar, scalar_to_i64, slice_tensor_value};

use crate::graph::CacheIndexExpr;

#[derive(Debug, Clone)]
pub struct TableIndexSelection {
    pub indices: Vec<usize>,
    pub is_scalar: bool,
}

#[derive(Debug, Clone)]
pub struct TensorIndexSelection {
    pub indices: Vec<usize>,
    pub is_scalar: bool,
}

#[derive(Debug, Clone)]
pub enum ResolvedCacheIndexExpr {
    Single(usize),
    Slice { start: Option<i64>, end: Option<i64> },
}

pub fn cache_index_is_single(expr: &CacheIndexExpr) -> bool {
    matches!(expr, CacheIndexExpr::Single(_))
}

