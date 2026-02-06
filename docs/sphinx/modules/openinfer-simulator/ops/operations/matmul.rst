matmul
======

**Category:** numerical

Signature
---------
`op matmul(...) >> out`

Arity
-----
- inputs: 2
- outputs: 1

Attributes
----------
- acc

Capabilities
------------
- broadcast: allow
- inplace: allow
- accumulate: allow

Devices
-------
- cpu, vulkan

DTypes
------
- normal: f8, bf16, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, bool, bitset, i1, i2, i4, u1, u2, u4
- accumulate: i1→i8, i1→i16, i1→i32, i1→i64, i2→i8, i2→i16, i2→i32, i2→i64, i4→i8, i4→i16, i4→i32, i4→i64, i8→i16, i8→i32, i8→i64, i16→i32, i16→i64, i32→i64, u1→u8, u1→u16, u1→u32, u1→u64, u2→u8, u2→u16, u2→u32, u2→u64, u4→u8, u4→u16, u4→u32, u4→u64, u8→u16, u8→u32, u8→u64, u16→u32, u16→u64, u32→u64

