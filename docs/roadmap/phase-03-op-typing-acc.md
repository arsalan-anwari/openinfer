# Phase 03 - Op Typing System and `acc` Redefinition

## Goal

Decouple accumulation semantics from output typing and move to explicit, policy-driven op typing validation.

## Compatibility Stance

This phase is a contract cleanup phase. Do not preserve legacy shorthand behavior when it conflicts with the target typing model.

## Scope

- Redefine op schema typing model:
  - valid input dtypes
  - valid output dtypes
  - accumulation policy (independent of output dtype)
- Upgrade DSL settings parser to support structured accumulation policy payloads.
- Enforce output variable dtype/shape compatibility against op policy.

## Design Decisions

1. **`acc` does not imply output type**.
2. **Output type is jointly constrained** by:
   - output variable declaration,
   - op output typing policy.
3. **Structured accumulation policy**
   - allow op-specific segmentation semantics (lane/block/group/etc.).
4. **Validation remains fail-fast and deterministic**.

## Primary Implementation Targets

- `openinfer-simulator/src/op_defs.rs`
- `openinfer-simulator/src/runtime/validation/ops.rs`
- `openinfer-simulator/src/runtime/op_runner.rs`
- `openinfer-simulator/ops.json`
- `openinfer-dsl/src/parsers/op.rs`
- `openinfer-dsl/src/validation/ops/mod.rs`

## Work Breakdown

1. Introduce schema representation for accumulation policy separate from output typing.
2. Update `ops.json` schema and parser logic.
3. Expand DSL attr parsing for richer policy values.
4. Update runtime validation to check:
   - output declaration compatibility,
   - policy validity per op.
5. Update dispatch keying only where needed for policy-resolved kernel variants.

## Test Plan

- **Schema tests**
  - op policy decode and static validation.
- **DSL tests**
  - valid and invalid `acc` policy syntax coverage.
- **Runtime validation tests**
  - output dtype mismatch rejection.
  - illegal policy rejection.
  - legal policy acceptance across modes.

## Acceptance Criteria

- `acc` semantics are policy-only, not implicit output type override.
- Output typing is explicit and op-constrained.
- Invalid type/policy combinations produce actionable validation errors.
- Existing ops with simple policies still work after migration.

## Risks and Mitigations

- Risk: complexity increase in DSL attr parsing.
  - Mitigation: clear grammar constraints and targeted parser tests.
- Risk: op schema migration churn.
  - Mitigation: staged migration with strict compile-time and runtime checks.

## Example `acc` Policy Shape

Instead of:

```dsl
op matmul(a, b, acc=i32) >> out;
```

support:

```dsl
op matmul(
  a,
  b,
  acc={
    lane: i16,
    block: i32,
    final: i32
  }
) >> out;
```

or list form for ops with custom segmentation:

```dsl
acc=[{scope:"lane", dtype:i16}, {scope:"group", dtype:i32}]
```

## Example Schema Extension (`ops.json` Concept)

```json
{
  "name": "matmul",
  "type_rule": { "kind": "schema_driven" },
  "typing": {
    "inputs": ["i8", "u8", "f16", "f32"],
    "outputs": ["i32", "f16", "f32"],
    "acc_policy": {
      "kind": "scoped_map",
      "scopes": ["lane", "block", "final"]
    }
  }
}
```

## Runtime Validation Pseudocode

```rust
let policy = parse_acc_policy(attrs)?;
ensure!(schema.acc_policy.validate(&policy), "invalid acc policy");
ensure!(schema.output_types.contains(output_decl.dtype), "output dtype forbidden");
ensure!(schema.output_shape_rule.validate(inputs, output_decl.shape), "output shape mismatch");
```

## Practical Migration Strategy

- Perform a one-shot schema and DSL migration to policy-structured `acc` semantics.
- Update existing models/tests to the canonical form in the same phase.
- Reject legacy shorthand forms once migration lands.
