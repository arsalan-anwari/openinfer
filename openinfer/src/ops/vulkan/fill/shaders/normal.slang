#include "../../shaders/common.slang"
#include "../../shaders/float_utils.slang"
#include "common.slang"

#ifndef HAS_F16
#define HAS_F16 0
#endif

#ifndef HAS_I64
#define HAS_I64 0
#endif

#ifndef HAS_U64
#define HAS_U64 0
#endif

#ifndef HAS_F64
#define HAS_F64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { FillPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    float value = asfloat(push.value_f32_bits);
    store_f32(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_f32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    float value = asfloat(push.value_f32_bits);
    store_f32(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_f16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    float value = asfloat(push.value_f32_bits);
    store_f16(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_f16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    float value = asfloat(push.value_f32_bits);
    store_f16(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    float value = asfloat(push.value_f32_bits);
    store_bf16(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_bf16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    float value = asfloat(push.value_f32_bits);
    store_bf16(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    float value = asfloat(push.value_f32_bits);
    store_f8(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_f8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    float value = asfloat(push.value_f32_bits);
    store_f8(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    int value = int(push.value_i64_lo);
    store_i32(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    int value = int(push.value_i64_lo);
    store_i32(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    int value = int(push.value_i64_lo);
    store_i16(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    int value = int(push.value_i64_lo);
    store_i16(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    int value = int(push.value_i64_lo);
    store_i8(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    int value = int(push.value_i64_lo);
    store_i8(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    uint value = push.value_u64_lo;
    store_u32(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    uint value = push.value_u64_lo;
    store_u32(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    uint value = push.value_u64_lo;
    store_u16_val(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    uint value = push.value_u64_lo;
    store_u16_val(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    uint value = push.value_u64_lo;
    store_u8_val(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    uint value = push.value_u64_lo;
    store_u8_val(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_bool_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    uint value = push.value_bool;
    store_bool_at(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_bool_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    uint value = push.value_bool;
    store_bool_at(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_bitset_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    uint value = push.value_u8;
    store_bitset_at(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_bitset_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    uint value = push.value_u8;
    store_bitset_at(out, idx, value, out_buf);
}

#if HAS_I64
int64_t fill_value_i64() {
    uint64_t bits = (uint64_t(push.value_i64_hi) << 32) | uint64_t(push.value_i64_lo);
    return int64_t(bits);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    int64_t value = fill_value_i64();
    store_i64(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    int64_t value = fill_value_i64();
    store_i64(out, idx, value, out_buf);
}
#endif

#if HAS_U64
uint64_t fill_value_u64() {
    return (uint64_t(push.value_u64_hi) << 32) | uint64_t(push.value_u64_lo);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    uint64_t value = fill_value_u64();
    store_u64(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    uint64_t value = fill_value_u64();
    store_u64(out, idx, value, out_buf);
}
#endif

#if HAS_F64
double fill_value_f64() {
    uint2 bits = uint2(push.value_f64_lo, push.value_f64_hi);
    return asdouble(bits);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    double value = fill_value_f64();
    store_f64(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_f64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[0];
    double value = fill_value_f64();
    store_f64(out, idx, value, out_buf);
}
#endif
