# Phase 07 - Stabilization, Conformance, and Release Readiness

## Goal

Lock down correctness, determinism, and baseline performance before the first stable foundation release.

## Scope

- Build a conformance matrix over:
  - op
  - input/output dtype
  - execution mode (`normal`/`packed`/`quantized`/`simulated`)
  - backend (CPU/Vulkan)
- Add integration workloads covering broader ML usage (not LLM-only).
- Add repeatable baseline benchmarks and guardrails.
- Finalize stable foundation documentation and migration notes.

## Design Decisions

1. CPU is the oracle backend for semantic correctness.
2. Vulkan parity is required for supported combinations (float tolerances documented).
3. CI must include deterministic replay and regression gates.
4. Stable release is blocked on conformance completeness, not only unit coverage.

## Primary Implementation Targets

- simulator tests and integration suites:
  - `openinfer-simulator/src/**`
  - `openinfer-simulator/examples/**`
- OINF fixture generation:
  - `openinfer-oinf/examples/**`
- docs:
  - `docs/roadmap/summary.md`
  - `docs/wiki/**`
  - `docs/sphinx/modules/openinfer-simulator/**`

## Work Breakdown

1. Define conformance matrix and expected verdicts.
2. Add deterministic replay tests and trace-based assertions.
3. Add integration models (classical ML + CV + quantized paths).
4. Add benchmark harness and threshold checks.
5. Publish stabilization report and release checklist.

## Test Plan

- **CI matrix**
  - backend x mode x dtype x op subsets.
- **Determinism tests**
  - repeat runs produce stable outputs/traces where expected.
- **Performance smoke tests**
  - baseline deltas with threshold alarms.
- **Compatibility tests**
  - ensure declared stable contracts are respected end-to-end.

## Acceptance Criteria

- Conformance matrix meets target completeness.
- No P0/P1 correctness regressions open.
- Determinism and validation diagnostics are stable.
- Stable release checklist is fully green.

## Risks and Mitigations

- Risk: CI explosion from matrix size.
  - Mitigation: tiered matrix (core per-PR, full nightly).
- Risk: benchmark noise.
  - Mitigation: controlled environment and rolling baseline window.

## Conformance Matrix Template (Example)

```text
op, mode, in_dtype, out_dtype, backend, expected
matmul, quantized, i8, i32, cpu, pass
matmul, quantized, i8, i32, vulkan, pass
add, packed, i4, i4, cpu, pass
add, packed, i4, i4, vulkan, pass
cast, normal, f32, i4, cpu, pass
cast, normal, f32, i4, vulkan, pass
```

## Determinism Test Sketch

```rust
#[test]
fn deterministic_replay_matmul_quantized() {
    let out1 = run_graph(seed=42);
    let out2 = run_graph(seed=42);
    assert_eq!(out1, out2, "replay mismatch");
}
```

## Benchmark Gate Example

```text
benchmark: matmul_i8_i32_cpu
baseline: 1.00x
gate: fail if slower than 1.15x baseline for 3 consecutive runs
```

## Release Checklist (Short Form)

- All phase acceptance criteria marked complete.
- Full nightly conformance matrix green.
- No unresolved high-severity correctness issues.
- Docs updated for stable contracts and non-backward-compatible changes.
