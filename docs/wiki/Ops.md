# Ops and DType Support

This page mirrors the current op catalog in `ops.json`. It lists the normal
dtype support and whether each op allows in-place, accumulation, and broadcast.
For exact accumulate dtype pairs and output typing rules, consult `ops.json` or
`res/ops_v2.json`.

| Op          | Dtypes (normal)                                                                                     | Attrs                        | Inplace | Accumulate | Broadcast |
| ----------- | --------------------------------------------------------------------------------------------------- | ---------------------------- | ------- | ---------- | --------- |
| add         | f8, bf16, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, bool, bitset, i1, i2, i4, u1, u2, u4 | acc                          | Yes     | Yes        | Yes       |
| mul         | f8, bf16, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, bool, bitset, i1, i2, i4, u1, u2, u4 | acc                          | Yes     | Yes        | Yes       |
| abs         | f8, bf16, f16, f32, f64, i1, i2, i4, i8, i16, i32, i64                                              | acc                          | Yes     | Yes        | No        |
| relu        | f8, bf16, f16, f32, f64, i4, i8, i16, i32, i64                                                      | alpha, clamp_max             | Yes     | No         | No        |
| matmul      | f8, bf16, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, bool, bitset, i1, i2, i4, u1, u2, u4 | acc                          | Yes     | Yes        | Yes       |
| is_finite   | f8, bf16, f16, f32, f64                                                                             | None                         | No      | No         | No        |
| fill        | f8, bf16, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, bool, bitset, i1, i2, i4, u1, u2, u4 | value                        | Yes     | No         | No        |
| sub         | f8, bf16, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | acc                          | Yes     | Yes        | Yes       |
| div         | f8, bf16, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i4, u4                               | div_by_zero_mask             | Yes     | No         | Yes       |
| floor_div   | i8, i16, i32, i64, u8, u16, u32, u64, i4, u4                                                        | div_by_zero_mask             | Yes     | No         | Yes       |
| rem         | i8, i16, i32, i64, u8, u16, u32, u64, i4, u4                                                        | None                         | Yes     | No         | Yes       |
| fma         | f8, f16, bf16, f32, f64                                                                             | None                         | Yes     | No         | No        |
| neg         | f8, f16, bf16, f32, f64, i8, i16, i32, i64, i1, i2, i4                                              | None                         | Yes     | No         | No        |
| sign        | f8, f16, bf16, f32, f64, i8, i16, i32, i64, i1, i2, i4                                              | None                         | Yes     | No         | No        |
| recip       | f8, f16, bf16, f32, f64                                                                             | div_by_zero_mask             | Yes     | No         | No        |
| min         | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | None                         | Yes     | No         | No        |
| max         | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | None                         | Yes     | No         | No        |
| clamp       | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | min, max                     | Yes     | No         | No        |
| floor       | f8, f16, bf16, f32, f64                                                                             | None                         | Yes     | No         | No        |
| ceil        | f8, f16, bf16, f32, f64                                                                             | None                         | Yes     | No         | No        |
| round       | f8, f16, bf16, f32, f64                                                                             | None                         | Yes     | No         | No        |
| trunc       | f8, f16, bf16, f32, f64                                                                             | None                         | Yes     | No         | No        |
| and         | i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4, bool                                  | None                         | Yes     | No         | No        |
| or          | i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4, bool                                  | None                         | Yes     | No         | No        |
| xor         | i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4, bool                                  | None                         | Yes     | No         | No        |
| not         | i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4, bool                                  | None                         | Yes     | No         | No        |
| shl         | i8, i16, i32, i64, u8, u16, u32, u64, i2, i4, u2, u4                                                | bits                         | Yes     | No         | No        |
| shr         | i8, i16, i32, i64, u8, u16, u32, u64, i2, i4, u2, u4                                                | bits                         | Yes     | No         | No        |
| popcount    | i8, i16, i32, i64, u8, u16, u32, u64, i2, i4, u2, u4                                                | None                         | No      | No         | No        |
| eq          | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4, bool         | None                         | No      | No         | No        |
| ne          | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4, bool         | None                         | No      | No         | No        |
| lt          | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | None                         | No      | No         | No        |
| le          | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | None                         | No      | No         | No        |
| gt          | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | None                         | No      | No         | No        |
| ge          | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | None                         | No      | No         | No        |
| filter      | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4, bool         | None                         | No      | No         | No        |
| is_nan      | f8, f16, bf16, f32, f64                                                                             | None                         | No      | No         | No        |
| is_inf      | f8, f16, bf16, f32, f64                                                                             | None                         | No      | No         | No        |
| is_neg      | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4                           | None                         | No      | No         | No        |
| sum_axis    | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | axes, keepdims, acc          | No      | Yes        | No        |
| mean_axis   | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | axes, keepdims, acc          | No      | Yes        | No        |
| prod_axis   | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | axes, keepdims, acc          | No      | Yes        | No        |
| max_axis    | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | axes, keepdims               | No      | No         | No        |
| min_axis    | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | axes, keepdims               | No      | No         | No        |
| argmax_axis | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | axis, keepdims, select_first | No      | No         | No        |
| argmin_axis | f8, f16, bf16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64, i1, i2, i4, u1, u2, u4               | axis, keepdims, select_first | No      | No         | No        |
| cast        | i8, i16, i32, i64, u8, u16, u32, u64, f8, f16, bf16, f32, f64, i1, i2, i4, u1, u2, u4               | to, rounding_mode, saturate  | No      | No         | No        |
