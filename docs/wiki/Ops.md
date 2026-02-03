# Ops and DType Support

This page mirrors the current op catalog in `ops.json`. It lists the normal
dtype support, input/output tensor formats, and whether each op allows in-place,
accumulation, and broadcast. For exact accumulate dtype pairs and output typing
rules, consult `ops.json` or `res/ops_v2.json`.

Dtypes are grouped into unpacked and packed forms for readability.

## Arithmetic Ops

| Op | Dtypes (normal) | Input | Output | Attrs | Inplace | Accumulate | Broadcast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| add | f8, bf16, f16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4<br>bool, bitset | Tensor[T] | Tensor[T] | None | Yes | Yes | Yes |
| mul | f8, bf16, f16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4<br>bool, bitset | Tensor[T] | Tensor[T] | None | Yes | Yes | Yes |
| abs | f8, bf16, f16, f32, f64<br>i8, i16, i32, i64<br>i1, i2, i4 | Tensor[T] | Tensor[T] | None | Yes | Yes | No |
| sub | f8, bf16, f16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[T] | None | Yes | Yes | Yes |
| div | f8, bf16, f16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i4, u4 | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[T] | div_by_zero_mask: T | Yes | No | Yes |
| floor_div | i8, i16, i32, i64, u8, u16, u32, u64<br>i4, u4 | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[T] | div_by_zero_mask: T | Yes | No | Yes |
| rem | i8, i16, i32, i64, u8, u16, u32, u64<br>i4, u4 | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[T] | None | Yes | No | Yes |
| fma | f8, f16, bf16, f32, f64 | a: Tensor[T]<br>b: Tensor[T]<br>c: Tensor[T] | y: Tensor[T] | None | Yes | No | No |
| neg | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64<br>i1, i2, i4 | x: Tensor[T] | y: Tensor[T] | None | Yes | No | No |
| recip | f8, f16, bf16, f32, f64 | x: Tensor[T] | y: Tensor[T] | div_by_zero_mask: T | Yes | No | No |

## Numerical Ops

| Op | Dtypes (normal) | Input | Output | Attrs | Inplace | Accumulate | Broadcast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| relu | f8, bf16, f16, f32, f64<br>i8, i16, i32, i64<br>i4 | Tensor[T] | Tensor[T] | alpha: T<br>clamp_max: T | Yes | No | No |
| matmul | f8, bf16, f16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4<br>bool, bitset | Tensor[T] | Tensor[T] | None | Yes | Yes | Yes |

## Rounding Ops

| Op | Dtypes (normal) | Input | Output | Attrs | Inplace | Accumulate | Broadcast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clamp | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | x: Tensor[T] | y: Tensor[T] | min: T<br>max: T| Yes | No | No |
| floor | f8, f16, bf16, f32, f64 | x: Tensor[T] | y: Tensor[T] | None | Yes | No | No |
| ceil | f8, f16, bf16, f32, f64 | x: Tensor[T] | y: Tensor[T] | None | Yes | No | No |
| round | f8, f16, bf16, f32, f64 | x: Tensor[T] | y: Tensor[T] | None | Yes | No | No |
| trunc | f8, f16, bf16, f32, f64 | x: Tensor[T] | y: Tensor[T] | None | Yes | No | No |

## Statistics Ops

| Op | Dtypes (normal) | Input | Output | Attrs | Inplace | Accumulate | Broadcast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sign | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64<br>i1, i2, i4 | x: Tensor[T] | y: Tensor[i8] | None | Yes | No | No |
| min | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[T] | None | Yes | No | No |
| max | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[T] | None | Yes | No | No |

## Bitwise Ops

| Op | Dtypes (normal) | Input | Output | Attrs | Inplace | Accumulate | Broadcast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| and | i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4<br>bool | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[T] | None | Yes | No | No |
| or | i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4<br>bool | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[T] | None | Yes | No | No |
| xor | i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4<br>bool | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[T] | None | Yes | No | No |
| not | i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4<br>bool | x: Tensor[T] | y: Tensor[T] | None | Yes | No | No |
| shl | i8, i16, i32, i64, u8, u16, u32, u64<br>i2, i4, u2, u4 | x: Tensor[T] | y: Tensor[T] | bits: u8 | Yes | No | No |
| shr | i8, i16, i32, i64, u8, u16, u32, u64<br>i2, i4, u2, u4 | x: Tensor[T] | y: Tensor[T] | bits: u8 | Yes | No | No |
| popcount | i8, i16, i32, i64, u8, u16, u32, u64<br>i2, i4, u2, u4 | x: Tensor[T] | y: Tensor[u8] | None | No | No | No |

## Comparison Ops

| Op | Dtypes (normal) | Input | Output | Attrs | Inplace | Accumulate | Broadcast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eq | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4<br>bool | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[bool] | None | No | No | No |
| ne | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4<br>bool | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[bool] | None | No | No | No |
| lt | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[bool] | None | No | No | No |
| le | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[bool] | None | No | No | No |
| gt | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[bool] | None | No | No | No |
| ge | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[bool] | None | No | No | No |

## Filter Ops

| Op | Dtypes (normal) | Input | Output | Attrs | Inplace | Accumulate | Broadcast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| is_finite | f8, bf16, f16, f32, f64 | Tensor[T] | Tensor[T] | None | No | No | No |
| filter | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4<br>bool | a: Tensor[T]<br>b: Tensor[T] | y: Tensor[T] | None | No | No | No |
| is_nan | f8, f16, bf16, f32, f64 | x: Tensor[T] | y: Tensor[bool] | None | No | No | No |
| is_inf | f8, f16, bf16, f32, f64 | x: Tensor[T] | y: Tensor[bool] | None | No | No | No |
| is_neg | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4 | x: Tensor[T] | y: Tensor[bool] | None | No | No | No |

## Reduction Ops

| Op | Dtypes (normal) | Input | Output | Attrs | Inplace | Accumulate | Broadcast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sum_axis | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | x: Tensor[T] | y: Tensor[T] | axes: i32[]<br>keepdims: bool | No | Yes | No |
| mean_axis | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | x: Tensor[T] | y: Tensor[T] | axes: i32[]<br>keepdims: bool | No | Yes | No |
| prod_axis | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | x: Tensor[T] | y: Tensor[T] | axes: i32[]<br>keepdims: bool| No | Yes | No |
| max_axis | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | x: Tensor[T] | y: Tensor[T] | axes: i32[]<br>keepdims: bool | No | No | No |
| min_axis | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | x: Tensor[T] | y: Tensor[T] | axes: i32[]<br>keepdims: bool | No | No | No |
| argmax_axis | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | x: Tensor[T] | y: Tensor[i64] | axis: i32<br>keepdims: bool<br>select_first: bool | No | No | No |
| argmin_axis | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | x: Tensor[T] | y: Tensor[i64] | axis: i32<br>keepdims: bool<br>select_first: bool | No | No | No |

## Casting Ops

| Op | Dtypes (normal) | Input | Output | Attrs | Inplace | Accumulate | Broadcast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cast | f8, f16, bf16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4 | x: Tensor[From] | y: Tensor[To] | to: dtype<br>rounding_mode: str<br>saturate: bool| No | No | No |

## Mutation Ops

| Op | Dtypes (normal) | Input | Output | Attrs | Inplace | Accumulate | Broadcast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fill | f8, bf16, f16, f32, f64<br>i8, i16, i32, i64, u8, u16, u32, u64<br>i1, i2, i4, u1, u2, u4<br>bool, bitset | Tensor[T] | Tensor[T] | value: T | Yes | No | No |
