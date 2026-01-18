# Progress

Checklist of features described in the README and current implementation status.

## Implemented

- [x] Rust DSL to Graph parsing and validation
- [x] Graph (de)serialization helpers and examples
- [x] Simulator execution with tracing and timing
- [x] Model loader for `.oinf` packages
- [x] CPU backend with basic ops (add, mul, abs, relu)
- [x] Vulkan backend with basic ops (add, mul, abs, relu)
- [x] Per-op broadcasting with CPU and Vulkan support
- [x] Vulkan shader manifest and embedded SPIR-V workflow
- [x] Reference attribute and custom attribute parsing support
- [x] Prefix tables in `volatile` and `constant`
- [x] Loops (loop blocks and loop-exit semantics)
- [x] Inplace kernels for faster simulation when possible
- [x] Cache operations beyond basic memory access
- [x] Prefixed/autodim cache ergonomics
- [x] Branching and yielding across blocks
- [x] Support for types universal and special types (see [types.md](types.md)) and upsampling

## DSL Gaps

Features described in the README DSL sections that are not fully supported yet.

- [ ] Barrier and explicit control dependency nodes

## In Progress / Planned

- [ ] Expanded op coverage beyond core arithmetic ops
- [ ] Synthesizer implementation (scheduling, fusion, memory planning)
- [ ] Device architecture JSON input for the synthesizer
- [ ] Analyzer and optimization passes
- [ ] Porting remaining kernels from C to Rust
