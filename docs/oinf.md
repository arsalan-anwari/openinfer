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
```

Tensors use only numeric/bool types (1-12). BITSET is for metadata only.

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
u32 dtype          (ValueType 1-12 only)
u32 ndim
u32 flags          (bit 0 = HAS_DATA)
u64 dims[ndim]
u64 data_nbytes
u64 data_offset
```

If HAS_DATA is 0, `data_offset` and `data_nbytes` are 0.

Tensor data blobs are raw little-endian values in row-major order. BOOL tensors
store one byte per element.

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

* `dataclass_to_oinf.py` converts a Python dataclass instance into a deterministic
  `.oinf` file (sizevars, metadata, and tensors are inferred from fields).
* `verify_oinf.py` validates the binary layout and prints a human-readable view,
  including summary statistics and histograms for tensors with data.

Typical usage:

```bash
python examples/python/simple_oinf.py
python examples/python/minimal_oinf.py
python verify_oinf.py res/simple_model.oinf
python verify_oinf.py res/minimal_model.oinf
```

### Create a Binary from a Dataclass

`dataclass_to_oinf.py` can serialize any Python dataclass instance into an OINF
file. You can pass a module path to a dataclass and optionally provide JSON data
for its fields.

Minimal example (matches `examples/python/minimal_oinf.py`):

```python
from dataclasses import dataclass
import numpy as np

from dataclass_to_oinf import TensorSpec, write_oinf

@dataclass
class MinimalModel:
    B: int
    a: TensorSpec

def build_minimal() -> MinimalModel:
    rng = np.random.default_rng(1)
    B = 1024
    a = rng.uniform(-1.0, 1.0, size=B).astype(np.float32)
    return MinimalModel(B=B, a=TensorSpec(a))

write_oinf(build_minimal(), "minimal_model.oinf")
```

Simple example (matches `examples/python/simple_oinf.py`):

```python
from dataclasses import dataclass
import numpy as np

from dataclass_to_oinf import TensorSpec, UninitializedTensor, write_oinf

@dataclass
class ExampleModel:
    D: int
    B: int
    a: TensorSpec
    x: TensorSpec
    W_0: TensorSpec
    mode: str
    y: UninitializedTensor
    kernel: TensorSpec

def build_example() -> ExampleModel:
    rng = np.random.default_rng(0)
    D = 128
    B = 1024
    a = rng.normal(size=B).astype(np.float16)
    x = np.array(10.35, dtype=np.float32)
    w0 = rng.normal(size=D).astype(np.float32)
    kernel = rng.integers(0, 256, size=(D, D), dtype=np.uint8)
    return ExampleModel(
        D=D,
        B=B,
        a=TensorSpec(a),
        x=TensorSpec(x),
        W_0=TensorSpec(w0, name="W.0"),
        mode="clamp_up",
        y=UninitializedTensor(dtype="i16", shape=()),
        kernel=TensorSpec(kernel),
    )

write_oinf(build_example(), "simple_model.oinf")
```

CLI example (module path + JSON payload):

```bash
python dataclass_to_oinf.py --input examples.python.simple_oinf:ExampleModel --output model.oinf
python dataclass_to_oinf.py --input my_pkg.my_model:MyModel --json data.json --output model.oinf
```

## Verifier Output Examples

`res/minimal_model.oinf` (values truncated):

```
B := 1024

a: f32[1024] = { 0.0123, -0.451, 0.998, 0.104, -0.731, ..., 0.882, -0.143, 0.221, -0.905, 0.314 }
- [nbytes: 4096, min: -0.999, max: 0.999, mean: 0.0021, median: 0.0043, std: 0.579]
- hist:
    [-0.999,-0.799):98
    [-0.799,-0.599):96
    ...
```

`res/simple_model.oinf` (values truncated):

```
D := 128
B := 1024

mode: str = "clamp_up"

W.0: f32[128] = { 0.48424, 1.61435, -0.782165, -0.0947963, 1.15624, ..., -0.646709, 0.947614, 0.625521, -0.300354, 0.897275 }
- [nbytes: 512, min: -3.19735, max: 2.8745, mean: 0.093444, median: 0.16931, std: 1.02064]
- hist:
    [-3.19735,-2.59016):1
    [-2.59016,-1.98298):2
    ...

a: f16[1024] = { 0.125732, -0.13208, 0.640625, 0.104919, -0.535645, ..., 1.37988, -1.17969, 0.509766, -1.0752, -0.334229 }
- [nbytes: 2048, min: -3.90039, max: 3.06641, mean: -0.0491846, median: -0.0691223, std: 0.971848]
- hist:
    [-3.90039,-3.20371):2
    ...

kernel: u8[128, 128] = {
{ 163, 255, 148, 186, 142, ..., 208, 23, 236, 196, 15 } ,
{ 200, 64, 246, 249, 250, ..., 171, 56, 243, 37, 201 } ,
...
}
- [nbytes: 16384, min: 0, max: 255, mean: 127.408, median: 128, std: 74.2236]
- hist:
    [0,25.5):1710
    ...

x: f32 = 10.35

y: i16[] -- uninitialized
```

## Verification and Printing

`verify_oinf.py` performs structural validation before printing:

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
