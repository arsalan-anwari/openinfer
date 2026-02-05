Types and DTypes
================

OpenInfer supports a broad set of dtypes across CPU and Vulkan backends.

Universal types
---------------

- f32 / f64
- i8 / i16 / i32 / i64
- u8 / u16 / u32 / u64
- bool
- bitset

Universal types are supported across the runtime and are available in both
graph declarations and op signatures.

Special tensor types
--------------------

- f8 / f16 / bf16
- i1 / i2 / i4
- u1 / u2 / u4
- t1 / t2 (reserved)

The special types are primarily used for quantized and packed workloads. These
types remain packed in storage, which reduces memory bandwidth and improves
cache locality.

Packed ranges
-------------

- i1: {-1, 0}
- i2: {-2, -1, 0, 1}
- i4: {-8 ... 7}
- u1: {0, 1}
- u2: {0, 1, 2, 3}
- u4: {0 ... 15}
- t1: {-1, 1}
- t2: {-1, 0, 1}

Backend behavior
----------------

CPU
~~~

- Supports all universal and special tensor types.
- Packed types are stored and operated on in packed form.

Vulkan
~~~~~~

- i64/u64 require `shader_int64`.
- f64 requires `shader_float64`.
- Unsupported types fall back to CPU with a warning.

When a type is unavailable on Vulkan, the dispatcher chooses the CPU kernel if
one exists. This keeps the graph executable but may affect performance.

Packed storage
--------------

Packed integer types are stored as bit-packed bytes in `.oinf` payloads and in
GPU buffers. Kernels operate on packed data directly instead of materializing
full-width tensors.

Example: declaring packed tensors
---------------------------------

.. code-block:: rust

   dynamic {
     mask: u1[B, D];
     sign: i1[B, D];
   }

This is common for masks, bitset activations, and compact boolean buffers.
