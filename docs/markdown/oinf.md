# OINF Binary Format

OINF (Open Infer Neural Format) is a single-file binary container for model data.
It stores size variables, metadata, and tensors in a deterministic, aligned layout.

This document describes the on-disk layout, encoding rules, and validation
requirements used by the loader and the Python tools.

## High-Level Layout

```
[FileHeader][SizeVarsTable][MetadataTable][TensorIndexTable][DataBlobs...]
```

Each table is an index that points to payload blobs in the data section. All
section starts and payloads are aligned to 8 bytes and padded with 0x00.

## Strings

Strings are restricted ASCII: `[A-Za-z0-9._-]`.

Encoding:

```
u32 length
bytes[length]
padding to 8 bytes
```

## FileHeader (fixed size)

Little-endian fields, then padding to 8 bytes:

```
magic[5] = "OINF\0"
u32 version = 1
u32 flags   = 0
u32 n_sizevars
u32 n_metadata
u32 n_tensors
u32 reserved = 0
u64 offset_sizevars
u64 offset_metadata
u64 offset_tensors
u64 offset_data
u64 file_size
```

## ValueType Tags

```
1  I8     5  U8     9  F16     12 BOOL
2  I16    6  U16   10  F32     13 BITSET
3  I32    7  U32   11  F64     14 STRING
4  I64    8  U64   15  NDARRAY
16 BF16  17 F8   18 I4   19 I2   20 I1
21 U4    22 U2     23 U1   24 T2   25 T1
```

Tensors use numeric/bool/bitset types (1-13, 16-25). BITSET can be used for
either metadata or tensor values.

## SizeVarsTable

Repeated `n_sizevars` times:

```
String name
u64 value
```

Names must be unique.

## MetadataTable

Repeated `n_metadata` times:

```
String key
u32 value_type
u32 value_flags = 0
u64 value_nbytes
u64 value_offset
```

Every metadata value is stored as a blob in the data section.

### Metadata Payloads

Scalar numeric types:

```
raw little-endian bytes of the scalar
```

BOOL:

```
u8 0 or 1
```

STRING:

```
String encoding (u32 length + bytes + padding)
```

BITSET:

```
u32 bit_count
u32 byte_count = ceil(bit_count/8)
u8  bytes[byte_count]   (LSB-first within each byte)
padding to 8 bytes
```

NDARRAY (metadata arrays):

```
u32 element_type   (ValueType, numeric/bool only)
u32 ndim
u64 dims[ndim]
raw packed data (row-major)
padding to 8 bytes
```

## TensorIndexTable

Repeated `n_tensors` times:

```
String name
u32 dtype          (ValueType 1-12, 16-25 only)
u32 ndim
u32 flags          (bit 0 = HAS_DATA)
u64 dims[ndim]
u64 data_nbytes
u64 data_offset
```

If HAS_DATA is 0, `data_offset` and `data_nbytes` are 0.

Tensor data blobs are raw little-endian values in row-major order. BOOL tensors
store one byte per element. BITSET tensors store one byte per element (u8
bitset value). Packed integer types (I1/I2/I4/U1/U2/U4/T1/T2) pack values into
bytes, LSB-first within each byte (e.g., I2 packs 4 elements per byte).
Scalars are encoded with `ndim = 0` and an empty dims list; they have a single
element in the data blob.

## Text Illustration

Uncompacted view (conceptual):

```
D := 128
B := 1024

a: f16[B] = { ... }
x: f32 = 10.35
W.0: f32[D] = { ... }
mode: str = "clamp_up"
y: i16
kernel: u8[D, D] = { ... }
```

## Python Tools

Two helper scripts live at the repository root:

* `openinfer-oinf/dataclass_to_oinf.py` converts a Python dataclass instance into a deterministic
  `.oinf` file (sizevars, metadata, and tensors are inferred from fields).
* `openinfer-oinf/verify_oinf.py` validates the binary layout and prints a human-readable view,
  including summary statistics and histograms for tensors with data.

Typical usage:

```bash
python examples/openinfer-oinf/mlp_regression_oinf.py
python examples/openinfer-oinf/quantized_linear_oinf.py
python openinfer-oinf/verify_oinf.py res/models/mlp_regression.oinf
python openinfer-oinf/verify_oinf.py res/models/quantized_linear.oinf
```

### Create a Binary from a Dataclass

`openinfer-oinf/dataclass_to_oinf.py` can serialize any Python dataclass instance into an OINF
file. You can pass a module path to a dataclass and optionally provide JSON data
for its fields.
If you need a scalar tensor (not metadata), wrap the value with `TensorSpec`
so it is emitted into the tensor table with `ndim = 0`.

Example (matches `examples/openinfer-oinf/mlp_regression_oinf.py`):

```python
from dataclasses import dataclass

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "openinfer-oinf"))

import numpy as np

from dataclass_to_oinf import SizeVar, TensorSpec, write_oinf

@dataclass
class MlpRegressionModel:
    B: SizeVar
    D: SizeVar
    H: SizeVar
    O: SizeVar
    w1: TensorSpec
    b1: TensorSpec
    w2: TensorSpec
    b2: TensorSpec

def build_model() -> MlpRegressionModel:
    rng = np.random.default_rng(1)
    B, D, H, O = 4, 16, 32, 8
    w1 = rng.normal(scale=0.2, size=(D, H)).astype(np.float32)
    b1 = rng.normal(scale=0.05, size=(H,)).astype(np.float32)
    w2 = rng.normal(scale=0.2, size=(H, O)).astype(np.float32)
    b2 = rng.normal(scale=0.05, size=(O,)).astype(np.float32)
    return MlpRegressionModel(
        B=SizeVar(B),
        D=SizeVar(D),
        H=SizeVar(H),
        O=SizeVar(O),
        w1=TensorSpec(w1),
        b1=TensorSpec(b1),
        w2=TensorSpec(w2),
        b2=TensorSpec(b2),
    )

write_oinf(build_model(), "mlp_regression.oinf")
```

CLI example (module path + JSON payload):

```bash
python openinfer-oinf/dataclass_to_oinf.py --input my_pkg.my_model:MyModel --json data.json --output model.oinf
```

## Verifier Output Example

Example output format (values truncated):

```
B := 4
D := 16

w1: f32[16, 32] = { 0.122, -0.044, ..., 0.203 }
- [nbytes: 2048, min: -0.812, max: 0.734, mean: 0.012, median: 0.009, std: 0.214]

b1: f32[32] = { 0.041, -0.012, ..., 0.007 }
```

## Verification and Printing

`openinfer-oinf/verify_oinf.py` performs structural validation before printing:

* checks magic/version, offsets, alignment, and file bounds
* validates string character set
* verifies tensor data size matches `numel * sizeof(dtype)`

When printing, it shows:

* sizevars as `NAME := VALUE`
* metadata values with explicit types
* tensors with a short preview (first 5/last 5), or `-- uninitialized`
* per-tensor statistics (min/max/mean/median/std) and a histogram summary

## Validation Checklist

Readers should validate:

* magic and version
* offsets are ascending and aligned
* referenced blobs are in-bounds
* string character set
* tensor byte sizes match `numel * sizeof(dtype)`
