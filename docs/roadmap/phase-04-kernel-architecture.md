# Phase 04 - Kernel Variant Architecture (CPU + Vulkan)

## Goal

Standardize kernel organization and dispatch semantics so packed, quantized, simulated, and native execution paths are explicit and scalable.

## Compatibility Stance

This phase standardizes the target kernel taxonomy. Avoid legacy dispatch-compat layers that obscure mode semantics.

## Scope

- Introduce consistent kernel variants:
  - `normal` (native dtype semantics)
  - `packed` (packed values treated as real packed-domain values)
  - `quantized` (packed/native values interpreted via quant metadata)
  - `simulated` (hardware-emulation/simulated dtype behavior)
- Extend dispatch keying to include interpretation mode.
- Generate Vulkan specialization dispatch artifacts with bounded explosion strategy.

## Design Decisions

1. **Interpretation mode is first-class dispatch input**.
2. **Dual Vulkan buffer path**
   - typed-buffer fast path for native normal kernels,
   - byte-address path retained for packed/simulated/bit-level needs.
3. **Generated specialization map** over monolithic hand-written switch trees.
4. **CPU is reference semantics per mode**.

## Primary Implementation Targets

- `openinfer-simulator/src/ops/cpu/**`
- `openinfer-simulator/src/ops/vulkan/**`
- `openinfer-simulator/src/ops/registry.rs`
- `openinfer-simulator/src/ops/vulkan/op_helpers.rs`
- `openinfer-simulator/generator/**`

## Work Breakdown

1. Define mode taxonomy in op dispatch core.
2. Reorganize/extend generated kernel stubs to include new variants.
3. Add Vulkan specialization-id generation (compact type/mode encoding).
4. Implement fallback kernels for unsupported rare combinations.
5. Add diagnostic tracing for resolved variant at runtime.

## Test Plan

- **Dispatch tests**
  - key -> kernel resolution correctness and completeness.
- **Parity tests**
  - CPU vs Vulkan output parity by op/dtype/mode.
- **Shader artifact tests**
  - specialization map generation and compile coverage.

## Acceptance Criteria

- Dispatch resolves mode-specific kernels deterministically.
- CPU/Vulkan variant behavior is consistent within documented tolerances.
- Kernel organization matches agreed variant taxonomy across backends.
- Unsupported combinations fail with explicit diagnostics.

## Risks and Mitigations

- Risk: Vulkan specialization combinatorics.
  - Mitigation: grouped variants + generated maps + fallback route.
- Risk: regression from large-scale kernel refactor.
  - Mitigation: incremental op-category migration with parity gates.

## Suggested Dispatch Key (Example)

```rust
pub enum KernelSemanticMode {
    Normal,
    Packed,
    Quantized,
    Simulated,
}

pub struct OpKeyV2 {
    pub kind: OpKind,
    pub mode: OpMode,               // normal/inplace/accumulate
    pub semantic: KernelSemanticMode,
    pub in_dtypes: Vec<DType>,
    pub out_dtype: DType,
}
```

## Vulkan Type-ID Encoding Example

```slang
// [31:24]=semantic mode, [23:16]=in0, [15:8]=in1, [7:0]=out
uint makeTypeMask(uint semantic, uint in0, uint in1, uint out0) {
    return (semantic << 24) | (in0 << 16) | (in1 << 8) | out0;
}
```

```slang
switch (PC.typeIdMask) {
    case TYPE_MATMUL_I8_I8_I32_QUANT: matmul_quantized<int8_t, int32_t>(...); break;
    case TYPE_MATMUL_I4_I4_I32_PACKED: matmul_packed_i4_i32(...); break;
    default: fallback_kernel(...); break;
}
```

## File Layout Suggestion

- CPU:
  - `kernels/normal.rs`
  - `kernels/packed.rs`
  - `kernels/quantized.rs`
  - `kernels/simulated.rs`
- Vulkan shaders:
  - `shaders/normal.slang`
  - `shaders/packed.slang`
  - `shaders/quantized.slang`
  - `shaders/simulated.slang`

## Handoff Notes For Another AI

- Refactor one op family at a time (`add`, `matmul`, then reductions).
- Add dispatch completeness tests before migrating next family.
- Keep temporary internal fallback kernels only as delivery scaffolding and remove them before phase completion.
