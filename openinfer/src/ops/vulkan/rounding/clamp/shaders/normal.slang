#include "../../../shaders/common.slang"
#include "../../../shaders/float_utils.slang"
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
[[vk::push_constant]] cbuffer Push { ClampPush push; }

float min_f32() { return asfloat(push.min_f32_bits); }
float max_f32() { return asfloat(push.max_f32_bits); }

int min_i32() {
    uint64_t bits = (uint64_t(push.min_i64_hi) << 32u) | uint64_t(push.min_i64_lo);
    return int(bits);
}

int max_i32() {
    uint64_t bits = (uint64_t(push.max_i64_hi) << 32u) | uint64_t(push.max_i64_lo);
    return int(bits);
}

uint min_u32() {
    uint64_t bits = (uint64_t(push.min_u64_hi) << 32u) | uint64_t(push.min_u64_lo);
    return uint(bits);
}

uint max_u32() {
    uint64_t bits = (uint64_t(push.max_u64_hi) << 32u) | uint64_t(push.max_u64_lo);
    return uint(bits);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float value = load_f32(a, idx, data);
    float clamped = min(max(value, min_f32()), max_f32());
    store_f32(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_f32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float value = load_f32(a, idx, data);
    float clamped = min(max(value, min_f32()), max_f32());
    store_f32(a, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_f16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float value = load_f16(a, idx, data);
    float clamped = min(max(value, min_f32()), max_f32());
    store_f16(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_f16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float value = load_f16(a, idx, data);
    float clamped = min(max(value, min_f32()), max_f32());
    store_f16(a, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float value = load_bf16(a, idx, data);
    float clamped = min(max(value, min_f32()), max_f32());
    store_bf16(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_bf16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float value = load_bf16(a, idx, data);
    float clamped = min(max(value, min_f32()), max_f32());
    store_bf16(a, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float value = load_f8(a, idx, data);
    float clamped = min(max(value, min_f32()), max_f32());
    store_f8(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_f8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float value = load_f8(a, idx, data);
    float clamped = min(max(value, min_f32()), max_f32());
    store_f8(a, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int value = load_i32(a, idx, data);
    int clamped = min(max(value, min_i32()), max_i32());
    store_i32(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int value = load_i32(a, idx, data);
    int clamped = min(max(value, min_i32()), max_i32());
    store_i32(a, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint value = load_u32(a, idx, data);
    uint clamped = min(max(value, min_u32()), max_u32());
    store_u32(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    uint value = load_u32(a, idx, data);
    uint clamped = min(max(value, min_u32()), max_u32());
    store_u32(a, idx, clamped, out_buf);
}

#if HAS_I64
int64_t min_i64() {
    return int64_t((uint64_t(push.min_i64_hi) << 32u) | uint64_t(push.min_i64_lo));
}
int64_t max_i64() {
    return int64_t((uint64_t(push.max_i64_hi) << 32u) | uint64_t(push.max_i64_lo));
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int64_t value = load_i64(a, idx, data);
    int64_t clamped = min(max(value, min_i64()), max_i64());
    store_i64(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int64_t value = load_i64(a, idx, data);
    int64_t clamped = min(max(value, min_i64()), max_i64());
    store_i64(a, idx, clamped, out_buf);
}
#endif

#if HAS_U64
uint64_t min_u64() {
    return (uint64_t(push.min_u64_hi) << 32u) | uint64_t(push.min_u64_lo);
}
uint64_t max_u64() {
    return (uint64_t(push.max_u64_hi) << 32u) | uint64_t(push.max_u64_lo);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint64_t value = load_u64(a, idx, data);
    uint64_t clamped = min(max(value, min_u64()), max_u64());
    store_u64(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    uint64_t value = load_u64(a, idx, data);
    uint64_t clamped = min(max(value, min_u64()), max_u64());
    store_u64(a, idx, clamped, out_buf);
}
#endif

#if HAS_F64
double min_f64() {
    uint2 bits = uint2(push.min_f64_lo, push.min_f64_hi);
    return asdouble(bits);
}
double max_f64() {
    uint2 bits = uint2(push.max_f64_lo, push.max_f64_hi);
    return asdouble(bits);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    double value = load_f64(a, idx, data);
    double clamped = min(max(value, min_f64()), max_f64());
    store_f64(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_f64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    double value = load_f64(a, idx, data);
    double clamped = min(max(value, min_f64()), max_f64());
    store_f64(a, idx, clamped, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int value = load_i16(a, idx, data);
    int clamped = min(max(value, min_i32()), max_i32());
    store_i16(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int value = load_i16(a, idx, data);
    int clamped = min(max(value, min_i32()), max_i32());
    store_i16(a, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint value = load_u16_val(a, idx, data);
    uint clamped = min(max(value, min_u32()), max_u32());
    store_u16_val(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    uint value = load_u16_val(a, idx, data);
    uint clamped = min(max(value, min_u32()), max_u32());
    store_u16_val(a, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int value = load_i8(a, idx, data);
    int clamped = min(max(value, min_i32()), max_i32());
    store_i8(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int value = load_i8(a, idx, data);
    int clamped = min(max(value, min_i32()), max_i32());
    store_i8(a, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint value = load_u8_val(a, idx, data);
    uint clamped = min(max(value, min_u32()), max_u32());
    store_u8_val(out, idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    uint value = load_u8_val(a, idx, data);
    uint clamped = min(max(value, min_u32()), max_u32());
    store_u8_val(a, idx, clamped, out_buf);
}
