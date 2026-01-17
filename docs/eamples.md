# Rust Examples

This list mirrors the Rust examples wired into `openinfer/Cargo.toml`.

- `examples/rust/minimal.rs`: Minimal graph with one op to show DSL structure.
- `examples/rust/graph_nodes_traverse.rs`: Walks graph nodes and prints metadata for inspection.
- `examples/rust/trace_example.rs`: Demonstrates simulator tracing and timing.
- `examples/rust/serialize.rs`: Serializes a graph to JSON.
- `examples/rust/deserialize.rs`: Deserializes a graph from JSON.
- `examples/rust/relu.rs`: Runs a small graph with Relu op.
- `examples/rust/multidim_broadcast.rs`: Exercises broadcasting behavior.
- `examples/rust/reference.rs`: Demonstrates model variable reference and lookup.
- `examples/rust/prefix_table.rs`: Uses prefix table access in non-persistent memory.
- `examples/rust/loop.rs`: Uses the DSL `loop` node to repeat ops.
- `examples/rust/worst_case.rs`: Stress-like example that builds a larger graph.
- `examples/rust/cache_scalar.rs`: Persistent scalar cache with increment/reset ops.
- `examples/rust/cache_table.rs`: Persistent table cache with indexed reads/writes.
- `examples/rust/cache_auto_dim.rs`: Persistent auto-dim cache growth driven by indices.
- `examples/rust/cache_weight_update.rs`: Updates cached weights across steps using ops.
- `examples/rust/cache_fixed_limit.rs`: Shows `@fixed` bounds triggering a cache error.
- `examples/rust/yield.rs`: Demonstrates yield/await concurrency across blocks.
