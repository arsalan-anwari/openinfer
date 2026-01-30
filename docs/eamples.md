# Rust Examples

This list mirrors the Rust examples wired into `openinfer/Cargo.toml`.

- `examples/openinfer/minimal.rs`: Minimal graph with one op to show DSL structure.
- `examples/openinfer/accumulate_packed.rs`: Packed dtype accumulation coverage.
- `examples/openinfer/branching_bad.rs`: Shows validation failure for bad branch graphs.
- `examples/openinfer/branching_good.rs`: Demonstrates valid branching and tracing.
- `examples/openinfer/cache_auto_dim.rs`: Persistent auto-dim cache growth driven by indices.
- `examples/openinfer/cache_fixed_limit.rs`: Shows `@fixed` bounds triggering a cache error.
- `examples/openinfer/cache_scalar.rs`: Persistent scalar cache with increment/reset ops.
- `examples/openinfer/cache_table.rs`: Persistent table cache with indexed reads/writes.
- `examples/openinfer/cache_weight_update.rs`: Updates cached weights across steps using ops.
- `examples/openinfer/deserialize.rs`: Deserializes a graph from JSON.
- `examples/openinfer/dtypes.rs`: Enumerates dtype coverage.
- `examples/openinfer/graph_nodes_traverse.rs`: Walks graph nodes and prints metadata for inspection.
- `examples/openinfer/loop.rs`: Uses the DSL `loop` node to repeat ops.
- `examples/openinfer/multidim_broadcast.rs`: Exercises broadcasting behavior with head/tail output formatting.
- `examples/openinfer/ops_accumulate_inplace.rs`: Validates CPU vs device outputs with drift-aware checks.
- `examples/openinfer/ops_broadcast_variants.rs`: Broadcast variants across ops.
- `examples/openinfer/ops_matrix.rs`: Matrix-focused op coverage and dtype checks.
- `examples/openinfer/ops_variants.rs`: Op variants and dtype coverage.
- `examples/openinfer/prefix_table.rs`: Uses prefix table access in non-persistent memory.
- `examples/openinfer/reference.rs`: Demonstrates model variable reference and lookup.
- `examples/openinfer/relu.rs`: Runs a small graph with Relu op.
- `examples/openinfer/serialize.rs`: Serializes a graph to JSON.
- `examples/openinfer/trace_example.rs`: Demonstrates simulator tracing and timing.
- `examples/openinfer/worst_case.rs`: Stress-like example that builds a larger graph.
- `examples/openinfer/yield.rs`: Demonstrates yield/await concurrency across blocks.
