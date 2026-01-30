# Rust Examples

This list mirrors the Rust examples wired into `openinfer/Cargo.toml`.

- `examples/openinfer/minimal.rs`: Minimal graph with one op to show DSL structure.
- `examples/openinfer/graph_nodes_traverse.rs`: Walks graph nodes and prints metadata for inspection.
- `examples/openinfer/trace_example.rs`: Demonstrates simulator tracing and timing.
- `examples/openinfer/serialize.rs`: Serializes a graph to JSON.
- `examples/openinfer/deserialize.rs`: Deserializes a graph from JSON.
- `examples/openinfer/relu.rs`: Runs a small graph with Relu op.
- `examples/openinfer/multidim_broadcast.rs`: Exercises broadcasting behavior with head/tail output formatting.
- `examples/openinfer/reference.rs`: Demonstrates model variable reference and lookup.
- `examples/openinfer/prefix_table.rs`: Uses prefix table access in non-persistent memory.
- `examples/openinfer/loop.rs`: Uses the DSL `loop` node to repeat ops.
- `examples/openinfer/worst_case.rs`: Stress-like example that builds a larger graph.
- `examples/openinfer/cache_scalar.rs`: Persistent scalar cache with increment/reset ops.
- `examples/openinfer/cache_table.rs`: Persistent table cache with indexed reads/writes.
- `examples/openinfer/cache_auto_dim.rs`: Persistent auto-dim cache growth driven by indices.
- `examples/openinfer/cache_weight_update.rs`: Updates cached weights across steps using ops.
- `examples/openinfer/cache_fixed_limit.rs`: Shows `@fixed` bounds triggering a cache error.
- `examples/openinfer/yield.rs`: Demonstrates yield/await concurrency across blocks.
- `examples/openinfer/ops_accumulate_inplace.rs`: Validates CPU vs device outputs with drift-aware checks.
- `examples/openinfer/ops_matrix.rs`: Matrix-focused op coverage and dtype checks.
- `examples/openinfer/f16_benchmark.rs`: Compares native vs simulated f16 performance (Vulkan).
