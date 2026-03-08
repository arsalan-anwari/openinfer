# Phase 03 - Op Typing System and Simple `acc` List

## Goal

Make accumulation typing easy in the DSL: users provide only a list of accumulator dtypes (`acc=[...]`), output dtype is taken from `>> out`, and unsupported combinations are rejected during graph validation.

## Compatibility Stance

This phase simplifies the contract and removes complex `acc` policy structures. Do not keep legacy structured `acc_policy` behavior.

## Scope

- Standardize DSL accumulation syntax to an ordered dtype list only:
  - `acc=[i16, i32]`
- Keep output typing independent of `acc` and resolved from output variable declaration (`>> out`).
- Remove `acc_policy` and other structured accumulation objects from `ops.json`.
- Validate op support for `(inputs, acc list, output)` combinations before execution.

## Design Decisions

1. **`acc` is only a dtype list**, not a policy object.
2. **Output dtype is deduced from `>> out`**, never from `acc`.
3. **No `acc_policy` fields in `ops.json`**.
4. **Unsupported combinations fail at graph validation** with actionable errors.
5. **Kernel functions should not hardcode user-side `acc` rules**; they receive validated types from the runtime.

## Primary Implementation Targets

- `openinfer-dsl/src/parsers/op.rs`
- `openinfer-dsl/src/validation/ops/mod.rs`
- `openinfer-simulator/src/runtime/validation/ops.rs`
- `openinfer-simulator/src/runtime/op_runner.rs`
- `openinfer-simulator/src/op_defs.rs`
- `openinfer-simulator/ops.json`

## Work Breakdown

1. Replace DSL `acc` parsing with list-only parsing (`acc=[dtype, ...]`).
2. Remove structured accumulation policy schema from `ops.json`.
3. Keep `ops.json` focused on core op typing metadata (inputs/outputs and other non-acc policy metadata).
4. Add validation step that checks whether each op supports the requested accumulation list for the resolved output dtype.
5. Ensure kernel invocation receives a normalized `Vec<DType>` for `acc` after validation.

## Test Plan

- **DSL tests**
  - accept `acc=[i16]`, `acc=[i16, i32]`
  - reject non-list/structured forms (`acc={...}`, scoped objects)
- **Validation tests**
  - reject unsupported `(op, in_dtypes, acc list, out_dtype)` combinations
  - preserve clear error messages that explain which combination failed
- **Runtime tests**
  - validated models execute with identical kernel behavior across accepted combinations

## Acceptance Criteria

- Users can express accumulation as `acc=[...]` without any policy object syntax.
- Output type is controlled only by output variable declaration (`>> out`).
- `ops.json` no longer contains `acc_policy`-style fields.
- Unsupported combinations fail during validation, before kernel execution.

## Risks and Mitigations

- Risk: temporary migration breakage from older `acc` syntax.
  - Mitigation: fail with explicit migration error pointing to list-only syntax.
- Risk: unclear meaning of multi-entry `acc` lists for new users.
  - Mitigation: document list semantics in API docs and reference from validation errors.

## DSL Example

```dsl
op matmul(a, b, acc=[i16, i32]) >> out_i32;
```

`out_i32` determines output dtype. The `acc` list describes the accumulation dtype sequence expected by the op contract.

## `ops.json` Direction (Simplified)

`ops.json` should not contain `acc_policy`, scoped accumulation objects, or policy engines. Accumulation support can be resolved by runtime/kernel capability checks used during graph validation.

## Validation Pseudocode

```rust
let acc_types = parse_acc_list(attrs)?;           // Vec<DType>
let out_dtype = resolve_output_dtype(output_var)?; // from >> out
ensure!(
    registry.supports(op_kind, in_dtypes, &acc_types, out_dtype),
    "unsupported accumulation combination for op"
);
```

## Practical Migration Strategy

- Move all models to list-only `acc=[...]` syntax.
- Remove structured accumulation parsing and schema paths in one phase.
- Keep validation strict and fail-fast so unsupported combos never reach kernels.
