abs
===

**Category:** arithmetic

Signature
---------
`op abs(...) >> out`

Arity
-----
- inputs: 1
- outputs: 1

Attributes
----------
- acc

Capabilities
------------
- broadcast: deny
- inplace: allow
- accumulate: allow

Devices
-------
- cpu, vulkan

DTypes
------
- normal: f8, bf16, f16, f32, f64, i1, i2, i4, i8, i16, i32, i64
- accumulate: i1→i8, i1→i16, i1→i32, i1→i64, i2→i8, i2→i16, i2→i32, i2→i64, i4→i8, i4→i16, i4→i32, i4→i64, i8→i16, i8→i32, i8→i64, i16→i32, i16→i64, i32→i64

