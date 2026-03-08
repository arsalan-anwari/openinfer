# OpenInfer Foundation Roadmap Summary

## Purpose

Provide a practical, independent phase sequence to harden OpenInfer into a comprehensive ML inferencing foundation (not only LLM-oriented), with strong typing, tensor semantics, and backend correctness.

## Compatibility Policy

This roadmap is a forward-foundation track. Phases are expected to converge on canonical contracts without backward-compatibility shims for legacy behavior.

## Phase Index

- Phase 00: `phase-00-contracts.md`
- Phase 01: `phase-01-tensor-core.md`
- Phase 02: `phase-02-oinf-quantization.md`
- Phase 03: `phase-03-op-typing-acc.md`
- Phase 04: `phase-04-kernel-architecture.md`
- Phase 05: `phase-05-layout-ops.md`
- Phase 06: `phase-06-dtype-range-correctness.md`
- Phase 07: `phase-07-stabilization.md`

## Recommended Execution Order

1. Phase 00 -> define hard contracts and acceptance gates.
2. Phase 01 -> redesign tensor core (all later phases depend on this).
3. Phase 02 and Phase 03 can run in parallel after Phase 01:
   - OINF quantization representation,
   - op typing + `acc` semantics.
4. Phase 04 depends on Phase 02 + 03 outputs.
5. Phase 05 and 06 can progress after Phase 01 (independent streams).
6. Phase 07 consolidates conformance, determinism, and release readiness.

## Key Architecture Outcomes

- Tensor representation supports real view semantics (`shape/strides/offset`).
- Packed storage and quantized interpretation are explicitly separated.
- Output typing and accumulation policy become deterministic and policy-driven.
- CPU/Vulkan kernel taxonomy is explicit (`normal/packed/quantized/simulated`).
- Layout ops are first-class and validated.
- Dtype boundaries and saturation behavior are centralized and correct.

## Cross-Phase Test Strategy

- Golden OINF fixtures across packed and quantized modes.
- Property tests for indexing and conversion correctness.
- CPU as semantic oracle; Vulkan parity with documented float tolerance.
- Conformance matrix: `op x dtype x mode x backend`.
- Deterministic validation and diagnostics for all invalid cases.

## Definition of Done (Foundation Stable)

- All phase acceptance criteria are met.
- Conformance matrix targets are green.
- No unresolved high-severity correctness gaps.
- Stable documentation reflects final contracts and behavior.

## Quick Start For A New AI Chat

If a new chat needs to execute this roadmap with minimal context, follow this sequence:

1. Read:
   - `docs/roadmap/phase-00-contracts.md`
   - `docs/roadmap/phase-01-tensor-core.md`
   - `docs/roadmap/phase-02-oinf-quantization.md`
   - `docs/roadmap/phase-03-op-typing-acc.md`
2. Implement Phase 01 completely (tests included) before touching Phase 04/05.
3. Implement Phase 02 and 03 in parallel if possible.
4. Use Phase 04 to unify kernel modes and dispatch.
5. Land Phase 05 and 06 with hard boundary/layout tests.
6. Finish with Phase 07 conformance + benchmark gates.

## Handoff Assumptions

- Backward compatibility with legacy behavior is not required for this stable foundation track.
- CPU remains semantic reference; Vulkan is parity target.
- Packed storage and quantized semantics must remain separate concepts.

## Minimal Code Fragments To Keep In Mind

```rust
// Tensor view model target
Tensor { data, shape, strides: Vec<isize>, offset_elems, quant: Option<QuantParams> }
```

```text
// OINF principle
packed != quantized
```

```rust
// Typing principle
output_dtype = validate(op_policy, output_decl, attrs)
```

# Done 
- [x] Phase 1
- [x] Phase 2
- [x] Phase 3
- [] Phase 4
- [] Phase 5
- [] Phase 6
- [] Phase 7