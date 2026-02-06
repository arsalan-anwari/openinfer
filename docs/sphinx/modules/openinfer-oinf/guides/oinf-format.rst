OINF Format
===========

OINF (Open Infer Neural Format) is a single-file binary container for model
data. It stores size variables, metadata, and tensors in a deterministic layout
with strict alignment rules. The runtime relies on this layout to lazily load
weights, resolve symbolic dimensions, and validate data integrity. This guide is
intentionally deep and concrete because the OINF format is central to how
OpenInfer operates: if the binary is wrong, execution will fail, but if the
binary is correct, everything else becomes deterministic and debuggable.

The format is designed to be:

- **Deterministic**: identical inputs produce identical binaries.
- **Random-access**: each tensor payload can be read independently.
- **Validated**: offsets, sizes, and types are checked before execution.
- **Friendly to memory mapping**: the runtime can lazily load tensors.

High-level layout and alignment
-------------------------------

OINF uses a fixed header and three index tables that point into a contiguous data
blob section. All sections are aligned to 8 bytes and padded with `0x00`.
Alignment is not optional: it is part of the contract and is validated by the
loader.

.. code-block:: text

   [FileHeader][SizeVarsTable][MetadataTable][TensorIndexTable][DataBlobs...]

You should think of the tables as *indexes* and the data section as the *payload
arena*. The tables never store values inline; they only store offsets and sizes.
This keeps the header compact and makes random access fast.

Strings and encoding rules
--------------------------

Strings are used for names (sizevars, metadata keys, tensor names). They use a
restricted ASCII character set: `[A-Za-z0-9._-]`. This is intentional: it keeps
parsing simple and ensures cross-platform compatibility.

String encoding:

.. code-block:: text

   u32 length
   bytes[length]
   padding to 8 bytes

The padding is required even for zero-length strings (though zero-length names
are invalid for identifiers).

FileHeader (fixed size)
-----------------------

The header is fixed size and uses little-endian encoding. It includes version,
counts, and offsets. The runtime validates the header before anything else.

.. code-block:: text

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

Offsets must be ascending and aligned. If any offset points outside the file,
the loader rejects the file immediately. The header is the authoritative source
for file integrity.

ValueType tags
--------------

The format uses numeric tags for data types. These tags are used both for
metadata and for tensors. The list is stable and should not be changed without
coordinated updates to the runtime and tooling.

.. code-block:: text

   1  I8     5  U8     9  F16     12 BOOL
   2  I16    6  U16   10  F32     13 BITSET
   3  I32    7  U32   11  F64     14 STRING
   4  I64    8  U64   15  NDARRAY
   16 BF16  17 F8   18 I4   19 I2   20 I1
   21 U4    22 U2     23 U1   24 T2   25 T1

Tensors use numeric/bool/bitset types. Metadata can additionally use `STRING`
and `NDARRAY`. Packed integer and ternary types are represented by their
specific tags and use bit-packed encoding in the data blobs.

SizeVarsTable
-------------

Sizevars encode symbolic dimensions. Each entry includes a string name and a
64-bit value. Names must be unique.

.. code-block:: text

   String name
   u64 value

These values are used to resolve dimensions like `B` or `D` in DSL variable
declarations. The runtime treats these as authoritative. If a graph refers to a
sizevar not present in the file, execution fails before any op runs.

MetadataTable
-------------

Metadata entries store scalars, strings, or arrays. Each entry records the type,
size, and offset into the data blob section.

.. code-block:: text

   String key
   u32 value_type
   u32 value_flags = 0
   u64 value_nbytes
   u64 value_offset

The payload is always stored in the data section. Metadata is *not* loaded
lazily in the same way as tensors; it is typically read when opening the model.

Metadata payloads:

- **Scalar numeric**: raw little-endian bytes of the scalar.
- **BOOL**: one byte (`0` or `1`).
- **STRING**: string encoding described above.
- **BITSET**: `u32 bit_count`, `u32 byte_count`, followed by packed bytes.
- **NDARRAY**: `element_type`, `ndim`, `dims`, and raw packed data.

This is deliberately strict: the reader can validate sizes without interpreting
the data. If `value_nbytes` does not match the expected size, the file is
invalid.

TensorIndexTable
----------------

The tensor index describes each tensor by name, dtype, dimensions, and payload
location. It is the central table the runtime uses for lazy loading.

.. code-block:: text

   String name
   u32 dtype          (ValueType 1-12, 16-25 only)
   u32 ndim
   u32 flags          (bit 0 = HAS_DATA)
   u64 dims[ndim]
   u64 data_nbytes
   u64 data_offset

If `HAS_DATA` is `0`, then `data_offset` and `data_nbytes` are `0`. This is a
valid representation of tensors that are declared but not initialized in the
model. In that case, the runtime uses DSL `@init` values or zeros.

Tensor payload encoding
-----------------------

Tensor data is stored in row-major order. Numeric types are stored as their
native little-endian bytes. BOOL tensors store one byte per element. BITSET
tensors store one byte per element (a `u8` bitset value).

Packed integer and ternary types (I1/I2/I4/U1/U2/U4/T1/T2) are stored in
bit-packed form. The packing is LSB-first within each byte. For example:

- I2 packs 4 elements per byte.
- I4 packs 2 elements per byte.
- I1 packs 8 elements per byte.

Scalars are encoded as `ndim = 0` with an empty dims list. Their payload is a
single element with the dtype’s size.

This encoding is consistent across CPU and Vulkan backends. Vulkan kernels use
helpers in `packed_utils.slang` to decode and encode packed values without
materializing large buffers.

Lazy loading in the runtime
---------------------------

The runtime uses memory mapping for `.oinf` files. The loader parses the header
and index tables, then stores offsets for each tensor. When an op needs a tensor,
the executor:

1. Resolves the tensor name and dtype.
2. Checks the `HAS_DATA` flag.
3. If data exists, reads only the requested byte range.
4. Caches the loaded tensor in memory.

This design keeps large models manageable and makes memory usage proportional to
the actual tensors used during execution.

Validation rules
----------------

The loader validates the file before execution. Key checks include:

- Magic/version correctness.
- Section offsets ascending and 8-byte aligned.
- Table entries within file bounds.
- String character set validity.
- Tensor sizes equal `numel * sizeof(dtype)` for non-packed types.
- Packed dtype sizes match the expected bit-packing.

If any validation fails, the loader returns an error and the graph does not run.
This prevents undefined behavior in kernels.

Example: conceptual binary layout
---------------------------------

The following is a conceptual view of a small model:

.. code-block:: text

   B := 4
   D := 16

   w1: f32[16, 32]
   b1: f32[32]
   mode: string = "clamp_up"

In the binary, `B` and `D` appear in the sizevars table, `mode` appears in the
metadata table, and `w1`/`b1` appear in the tensor index table. The actual data
for `w1` and `b1` is stored in the data blob section at the offsets specified in
the tensor index table.

Python tooling and examples
---------------------------

The Python tooling is the easiest way to produce valid `.oinf` files. The
primary entry points are:

- `openinfer-oinf/dataclass_to_oinf.py`
- `openinfer-oinf/verify_oinf.py`

Example usage:

.. code-block:: bash

   python openinfer-oinf/examples/mlp_regression_oinf.py
   python openinfer-oinf/verify_oinf.py openinfer-oinf/res/models/mlp_regression.oinf

The verifier prints a summary with statistics and also validates all offsets and
sizes. Use it whenever you generate a new model.

Debugging corrupted files
-------------------------

If verification fails, start by checking:

1. The header offsets and sizes.
2. String lengths and padding.
3. Tensor payload size calculations.
4. Packed dtype byte counts.

The most common errors are off-by-one padding mistakes or incorrect byte counts
for packed types. The verifier is strict because the runtime assumes the binary
is valid. When in doubt, generate a small model and compare its binary layout
against the expected structure.

Worked example: alignment and offsets
----------------------------------------

Consider a small model with two tensors: `x: f32[4]` and `y: u8[8]`, and one
metadata string `mode = "fast"`. The file layout will look like:

1. FileHeader
2. SizeVarsTable (possibly empty)
3. MetadataTable (one entry)
4. TensorIndexTable (two entries)
5. DataBlobs: string payload, `x` payload, `y` payload

Each section starts at an 8-byte boundary. If the string payload is 4 bytes, it
is padded to 8. The `x` payload is 16 bytes and already aligned. The `y` payload
is 8 bytes, also aligned. The offsets recorded in the tables must point to the
aligned payload start.

This example illustrates the key rule: every payload is aligned, and the tables
only reference aligned offsets. If you forget the padding, the verifier will
catch it, but the runtime would otherwise interpret wrong data.

Packed dtype sizing rules
-------------------------

Packed types use bit-level storage. The number of bytes is:

- `ceil(numel * bits_per_element / 8)`

For example:

- I1 with 9 elements -> `ceil(9 / 8) = 2` bytes.
- I2 with 9 elements -> `ceil(18 / 8) = 3` bytes.
- I4 with 9 elements -> `ceil(36 / 8) = 5` bytes.

Because the payload size is derived from bit counts, off-by-one errors are
common when writing generators. Always compute size using integer arithmetic
with rounding up, and always verify using `verify_oinf.py`.

Compatibility and versioning
----------------------------

The `version` field in the header is the compatibility gate. The runtime will
reject unknown versions. If you plan to extend the format, you must:

1. Add a new version number.
2. Update the loader to accept and parse it.
3. Update the Python tooling to emit the new version.

Do not change the meaning of existing fields without bumping the version. The
format is meant to be stable and deterministic across releases.

Security and robustness notes
-----------------------------

The OINF loader validates offsets and sizes to prevent out-of-bounds reads.
However, you should still treat `.oinf` files as untrusted input in external
tools. If you build custom tooling, follow the same validation checklist:

- Verify file size before reading any offsets.
- Validate all offsets and sizes.
- Reject malformed strings or unknown types.
- Avoid loading large payloads unless necessary.

Where to go next
----------------

- `API/openinfer-oinf` for the Python tooling interface.
- `Serialization` for how graphs are serialized to JSON.
- `Memory Model` for how persistent tensors are used by the runtime.

