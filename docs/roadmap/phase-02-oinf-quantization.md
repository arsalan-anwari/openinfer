# Phase 02 - OINF v2 Quantization Model

## Goal

Evolve OINF to represent quantization explicitly (per tensor or per channel) while preserving deterministic loading and strict validation.

## Scope

- Bump format version for quantization-capable files.
- Extend Python authoring path to encode quant metadata.
- Extend Rust loader to parse and attach quant metadata to runtime tensors.
- Keep packed-only (non-quantized) tensors supported.

## Design Decisions

1. **Versioned format change** (v2) instead of overloading v1 behavior.
2. **Per-tensor metadata association must be explicit**, not inferred from dtype.
3. **Quantization schema**
   - `scheme`: symmetric/asymmetric
   - `scale`: per-tensor or per-channel
   - `zero_point`: optional, per-tensor or per-channel
4. **Validation-first parser**
   - shape/axis/value-count checks for per-channel metadata.

## Primary Implementation Targets

- `openinfer-oinf/oinf_encoder.py`
- `openinfer-oinf/oinf_verify.py`
- `openinfer-simulator/src/runtime/model_loader.rs`
- `docs/sphinx/modules/openinfer-oinf/guides/oinf-format.rst`

## Work Breakdown

1. Introduce quant-capable `TensorSpec` contract in Python API.
2. Encode quant metadata in OINF v2 layout.
3. Extend verifier output and validation logic for quant sections.
4. Add loader parsing and runtime mapping to tensor quant metadata.
5. Ensure v1/v2 behavior is explicit and deterministic.

## Test Plan

- **Golden fixtures**
  - packed non-quantized
  - quantized per-tensor
  - quantized per-channel
  - asymmetric with zero-point
- **Verifier tests**
  - malformed axis/length mismatch must fail.
  - unsupported quant combinations must fail.
- **Loader tests**
  - quant metadata roundtrip and attachment to tensors.
  - v1 files still load according to declared support policy.

## Acceptance Criteria

- OINF v2 files can encode and verify quant metadata robustly.
- Loader surfaces quant metadata without ambiguity.
- Packed non-quantized behavior remains available.
- Validation catches malformed quant payloads before execution.

## Risks and Mitigations

- Risk: ambiguous backward behavior between v1 and v2.
  - Mitigation: strict version gate + explicit loader error messages.
- Risk: metadata bloat for per-channel quant tensors.
  - Mitigation: compact encoding and alignment-aware payload design.

## Proposed Python API Example

```python
from dataclasses import dataclass
from oinf_encoder import TensorSpec, QuantParams, QuantScale, QuantZeroPoint

@dataclass
class Model:
    w_q: TensorSpec
    x_u8: TensorSpec

model = Model(
    w_q=TensorSpec(
        data=w_int8,
        dtype="i8",
        quant=QuantParams(
            scheme="symmetric",
            scale=QuantScale.per_channel(axis=0, values=[...]),
            zero_point=None,
        ),
    ),
    x_u8=TensorSpec(
        data=x_uint8,
        dtype="u8",
        quant=QuantParams(
            scheme="asymmetric",
            scale=QuantScale.per_tensor(0.125),
            zero_point=QuantZeroPoint.per_tensor(128),
        ),
    ),
)
```

## Proposed OINF v2 Tensor Entry Sketch

```text
String name
u32 dtype
u32 ndim
u32 flags      (bit0 HAS_DATA, bit1 HAS_QUANT)
u64 dims[ndim]
u64 data_nbytes
u64 data_offset
u64 quant_nbytes   (0 if no quant)
u64 quant_offset   (0 if no quant)
```

## Loader Mapping Example (Rust Pseudocode)

```rust
let tensor = tensor_value_from_bytes(info, payload)?;
let quant = if info.has_quant {
    Some(parse_quant_params(mmap, info.quant_offset, info.quant_nbytes)?)
} else {
    None
};
tensor.set_quant(quant);
```

## Execution Guidance For Another AI

- Prefer adding v2 first-class support before touching dequant behavior in kernels.
- Keep verifier strictness high; malformed quant sections should fail before runtime execution.
- Add golden files early; use them as hard fixtures across encoder, verifier, loader.
