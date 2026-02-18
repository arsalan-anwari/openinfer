# Phase 06 - DType Range Correctness and Saturation Semantics

## Goal

Eliminate ambiguous or incorrect boundary behavior for packed/native/simulated dtypes and centralize conversion semantics.

## Scope

- Define canonical min/max for every dtype and interpretation mode.
- Replace ad-hoc casting/clamping logic with shared checked/saturating helpers.
- Fix packed signed boundaries and enforce them consistently in Python and Rust.

## Design Decisions

1. **Single source of truth** for dtype limits.
2. **Semantics explicit by operation**
   - checked conversion,
   - saturating conversion,
   - wrapping conversion (only where explicitly intended).
3. **Packed signed correctness is mandatory**
   - e.g. `i4` range is `[-8, 7]`, never `[-8, 8]`.

## Primary Implementation Targets

- `openinfer-simulator/src/tensor/scalar.rs`
- `openinfer-simulator/src/tensor/value.rs`
- casting/rounding kernels under:
  - `openinfer-simulator/src/ops/cpu/casting/**`
  - `openinfer-simulator/src/ops/cpu/rounding/**`
  - `openinfer-simulator/src/ops/vulkan/casting/**`
  - `openinfer-simulator/src/ops/vulkan/rounding/**`
- `openinfer-oinf/oinf_encoder.py`
- `openinfer-oinf/oinf_verify.py`

## Work Breakdown

1. Add centralized dtype-limit helpers for runtime and Python tooling.
2. Update packed scalar construction/conversion logic to enforce canonical bounds.
3. Ensure cast/rounding ops call shared conversion policy helpers.
4. Add explicit test vectors for edge and out-of-range behavior.

## Test Plan

- **Boundary tests per dtype**
  - min, max, min-1, max+1 paths.
- **Policy tests**
  - saturate vs wrap vs checked behavior coverage.
- **Fuzz tests**
  - random conversion chains preserve declared semantics.
- **Cross-language consistency**
  - Python encoder checks align with Rust loader/runtime assumptions.

## Acceptance Criteria

- All dtype boundaries are correct and centralized.
- No op uses silent ad-hoc boundary behavior.
- Packed boundary bugs (including `i4`) are demonstrably fixed by tests.

## Risks and Mitigations

- Risk: behavior changes in legacy edge cases.
  - Mitigation: document intentional semantic changes in release notes.
- Risk: duplicate boundary logic reappears.
  - Mitigation: lint/test guardrails against local hardcoded limits.

## Canonical Range Table (Core Examples)

```text
i1:  [-1, 0]
i2:  [-2, 1]
i4:  [-8, 7]
u1:  [0, 1]
u2:  [0, 3]
u4:  [0, 15]
t1:  {-1, +1}
t2:  {-1, 0, +1}
```

## Shared Conversion Helper Example

```rust
pub enum ConvertPolicy {
    Checked,
    Saturating,
    Wrapping,
}

pub fn convert_i32_to_i4(v: i32, policy: ConvertPolicy) -> Result<I4> {
    match policy {
        ConvertPolicy::Checked => {
            ensure!((-8..=7).contains(&v), "out of range for i4");
            Ok(I4::from_i8(v as i8))
        }
        ConvertPolicy::Saturating => Ok(I4::from_i8(v.clamp(-8, 7) as i8)),
        ConvertPolicy::Wrapping => Ok(I4::from_i8(v as i8)),
    }
}
```

## Python Encoder Guard Example

```python
def check_i4_range(arr: np.ndarray) -> None:
    if arr.size and (int(arr.min()) < -8 or int(arr.max()) > 7):
        raise OinfError("Values out of range for i4 [-8, 7]")
```

## Edge-Case Test Matrix Template

```text
dtype=i4, policy=checked:    -9 fail, -8 ok, 7 ok, 8 fail
dtype=i4, policy=saturating: -9 -> -8, 8 -> 7
dtype=u2, policy=checked:    -1 fail, 0 ok, 3 ok, 4 fail
```
