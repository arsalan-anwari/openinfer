# Supported Ops

---

| Op name   | Dtypes | Input | Output | Attributes | Inplace | Accumulation | Broadcast | Description |
|-----------|--------|-------|--------|------------|----------|--------------|-----------|-------------|
| add       | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[bool, bitset] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | Yes | Elementwise add |
| mul       | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[bool, bitset] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | Yes | Elementwise multiply |
| matmul    | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[bool, bitset] | a: Tensor[T], b: Tensor[T] | y: Tensor[T] |  | Yes | Yes | Yes | N-D matmul (dot product) |
| abs       | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64] | a: Tensor[T] | y: Tensor[T] |  | Yes | Yes | No | Elementwise absolute value |
| relu      | [f8, bf16, f16, f32, f64]<br>[i4, i8, i16, i32, i64] | a: Tensor[T] | y: Tensor[T] | alpha: T, clamp_max: T | Yes | No | No | Leaky ReLU with clamp |
| is_finite | [f8, bf16, f16, f32, f64] | a: Tensor[T] | y: bool | None | No | No | No | True if all elements are finite |
| fill      | [f8, bf16, f16, f32, f64]<br>[i1, i2, i4, i8, i16, i32, i64]<br>[u1, u2, u4, u8, u16, u32, u64]<br>[bool, bitset] | a: Tensor[T] | y: Tensor[T] | value: T | Yes | No | No | Fill output with a scalar literal |


