# Progress

Checklist of features described in the README and current implementation status.

## Implemented

- [x] Rust DSL parsing and validation
- [x] Graph (de)serialization helpers and examples
- [x] Simulator execution with tracing and timing
- [x] Model loader for `.oinf` packages
- [x] Core ops on CPU and Vulkan (add, mul, abs, relu, matmul, is_finite, fill)
- [x] Broadcast support for add/mul/matmul on CPU and Vulkan
- [x] Vulkan shader manifest and embedded SPIR-V workflow
- [x] Prefix tables in `volatile` and `constant`
- [x] Loops and branching control flow
- [x] In-place kernels and accumulation output reuse
- [x] Cache operations, prefixed tables, and auto-dim growth
- [x] Yield/await scheduling across blocks
- [x] Barrier/dep/transfer nodes
- [x] Packed integer and low-bit float handling
- [x] Vulkan CPU fallback for missing int64/float64 features

## In Progress / Planned

- [ ] Expanded op coverage beyond core arithmetic ops
- [ ] Synthesizer implementation (scheduling, fusion, memory planning)
- [ ] Analyzer and optimization passes
- [ ] Device architecture JSON input for reproducible synthesis
- [ ] Additional backend-specific optimizations and kernels
