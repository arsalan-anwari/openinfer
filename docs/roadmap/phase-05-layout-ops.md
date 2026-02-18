# Phase 05 - First-Class Layout and View Ops

## Goal

Make tensor layout manipulation explicit, validated, and reusable as standalone ops instead of hidden side effects in compute ops.

## Scope

Add first-class ops:

- `reshape` / `view`
- `transpose` / `permute`
- `slice` / `narrow`
- `select` (single index on axis)
- `squeeze`
- `unsqueeze`
- `expand` (broadcast view with stride 0 semantics)
- optional `as_strided` (power-user path with strict bounds checks)

## Design Decisions

1. Layout ops are metadata transformations first; no copy unless explicitly required.
2. Every op enforces legality rules up front (shape compatibility, bounds, rank constraints).
3. `as_strided` is optional and guarded by strict validation due to high misuse risk.

## Primary Implementation Targets

- `openinfer-simulator/ops.json`
- `openinfer-simulator/src/runtime/validation/**`
- `openinfer-simulator/src/runtime/op_runner.rs`
- new CPU/Vulkan op modules in existing category layout.

## Work Breakdown

1. Add schema entries for all layout/view ops.
2. Implement validation rules and canonical shape/stride derivation.
3. Implement CPU kernels as metadata transformations.
4. Add Vulkan support where applicable (metadata-only path can remain CPU-routed if needed initially).
5. Ensure graph typing and transfer semantics remain consistent.

## Test Plan

- **View algebra tests**
  - compose/reverse operations where mathematically expected.
- **Shape legality tests**
  - invalid reshape/slice/select/expand must fail clearly.
- **Non-contiguous tests**
  - downstream arithmetic ops consume produced views correctly.
- **Backend consistency tests**
  - parity for supported backend paths.

## Acceptance Criteria

- All listed layout ops are explicit and validated.
- Metadata-only transformations do not silently copy.
- Generated views are safe, bounds-checked, and interoperable with existing ops.

## Risks and Mitigations

- Risk: hidden assumptions in existing ops about contiguous tensors.
  - Mitigation: add contiguous fast-path + generic strided path tests.
- Risk: `as_strided` misuse leading to invalid memory mappings.
  - Mitigation: strict validation and optional feature gate initially.

## Example Semantics (Pseudo-API)

```rust
// reshape/view: same storage, different logical shape (compatible numel)
let y = x.view([B, H, W, C])?;

// permute: reorder shape + strides
let z = y.permute([0, 3, 1, 2])?;

// slice/narrow: adjust offset + one dim size
let s = z.narrow(axis=2, start=8, len=16)?;

// expand: stride=0 on expanded dims
let e = s.expand([B, C, 16, 16])?;
```

## Example Validation Rules

```rust
fn validate_reshape(old_shape: &[usize], new_shape: &[usize]) -> Result<()> {
    ensure!(numel(old_shape) == numel(new_shape), "reshape changes element count");
    Ok(())
}

fn validate_permute(rank: usize, order: &[usize]) -> Result<()> {
    ensure!(order.len() == rank, "permute rank mismatch");
    ensure!(is_permutation(order, rank), "permute indices must be unique");
    Ok(())
}
```

## Suggested `ops.json` Additions (Concept)

```json
{
  "name": "reshape",
  "kind": "reshape",
  "attrs": ["shape"],
  "type_rule": { "kind": "same_as_input", "index": 0 },
  "broadcast": "deny",
  "inplace": "deny",
  "accumulate": "deny"
}
```

## Practical Rollout

- Implement validation + CPU metadata kernels first.
- Route Vulkan for metadata-only ops through CPU path initially if needed.
- Add direct Vulkan paths later where it provides measurable benefit.
