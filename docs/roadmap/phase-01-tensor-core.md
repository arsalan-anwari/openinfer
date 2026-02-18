# Phase 01 - Tensor Core Redesign (Views + Quant Metadata)

## Goal

Replace the simplified tensor model with a robust view-capable representation that supports future layout ops, quantization metadata, and backend-safe indexing.

## Scope

- Introduce canonical tensor view metadata:
  - `shape: Vec<usize>`
  - `strides: Vec<isize>` (logical element strides)
  - `offset_elems: usize`
- Add optional quantization metadata on tensor values:
  - `QuantScheme`, `QuantScale`, `QuantZeroPoint`, `QuantParams`
- Keep packed storage and quantization semantics separate.
- Add strict bounds validation for all view construction paths.

## Design Decisions

1. **Signed strides are required** to support reverse/advanced views.
2. **Offset is explicit** so slice/select/narrow can be metadata-only.
3. **Quant metadata is optional and non-invasive**:
   - no automatic dequant in tensor container,
   - kernels decide interpretation policy.
4. **Packed tensors remain physically packed**:
   - no auto-unpack into widened storage.

## Primary Implementation Targets

- `openinfer-simulator/src/tensor/tensor.rs`
- `openinfer-simulator/src/tensor/shape.rs`
- `openinfer-simulator/src/tensor/value.rs`
- Any helper modules that assume unsigned strides or offset-free indexing.

## Work Breakdown

1. Add new tensor/view metadata structures and constructors.
2. Add shape/stride/offset validation utilities:
   - compute legal logical index range,
   - check storage coverage does not exceed backing allocation.
3. Update index and view methods to use signed stride math.
4. Thread quant metadata through `TensorValue` wrappers.
5. Update zero/clone/view helpers to preserve metadata consistently.

## Test Plan

- **Property tests**
  - logical index -> storage offset mapping for random valid views.
- **Unit tests**
  - contiguous and non-contiguous indexing equivalence.
  - negative stride correctness.
  - non-zero offset correctness.
  - invalid view rejection (out-of-bounds, inconsistent metadata).
- **Regression tests**
  - current existing ops still execute with contiguous tensors.

## Acceptance Criteria

- View metadata supports signed strides + explicit offset.
- Invalid views fail deterministically with clear errors.
- Quant metadata can be attached without changing non-quant behavior.
- Existing contiguous workloads remain correct.

## Risks and Mitigations

- Risk: latent assumptions in kernels about positive/contiguous strides.
  - Mitigation: gate non-contiguous path behind explicit validation until all ops are upgraded.
- Risk: packed tensor length assumptions break with offset math.
  - Mitigation: centralize storage-length and offset-range helpers for packed dtypes.

## Suggested Data Model (Example)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantScheme {
    Symmetric,
    Asymmetric,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantScale {
    PerTensor(f32),
    PerChannel { axis: usize, values: Vec<f32> },
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantZeroPoint {
    PerTensor(i32),
    PerChannel { axis: usize, values: Vec<i32> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantParams {
    pub scheme: QuantScheme,
    pub scale: QuantScale,
    pub zero_point: Option<QuantZeroPoint>,
}

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset_elems: usize,
    pub quant: Option<QuantParams>,
}
```

## Bounds Validation Example

```rust
fn validate_view(shape: &[usize], strides: &[isize], offset: usize, storage_len: usize) -> Result<()> {
    // Compute min/max reachable element index from logical index-space.
    // min_idx/max_idx are relative to offset.
    let (min_rel, max_rel) = reachable_index_range(shape, strides)?;
    let min_abs = offset as isize + min_rel;
    let max_abs = offset as isize + max_rel;

    ensure!(min_abs >= 0, "negative reachable storage index");
    ensure!((max_abs as usize) < storage_len, "view exceeds storage bounds");
    Ok(())
}
```

## Migration Notes For Another AI

- First update tensor internals and tests before touching many ops.
- Keep compatibility shim methods (`shape()`, `strides()`) while refactoring call sites.
- Add temporary helper wrappers for old unsigned-stride call paths and delete after Phase 05 lands.
