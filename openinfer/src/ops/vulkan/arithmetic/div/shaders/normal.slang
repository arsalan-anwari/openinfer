#include "../../../shaders/common.slang"
#include "../../../shaders/float_utils.slang"
#include "common.slang"



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
[[vk::push_constant]] cbuffer Push { DivPush push; }

float mask_f32() { return asfloat(push.mask_f32_bits); }

[numthreads(256, 1, 1)]
[shader("compute")]
void div_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    float lhs = load_f32(a, idx, data);
    float rhs = load_f32(b, idx, data);
    float result = (rhs == 0.0f) ? mask_f32() : (lhs / rhs);
    store_f32(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_f32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    float lhs = load_f32(a, idx, data);
    float rhs = load_f32(b, idx, data);
    float result = (rhs == 0.0f) ? mask_f32() : (lhs / rhs);
    store_f32(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_f16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    float lhs = load_f16(a, idx, data);
    float rhs = load_f16(b, idx, data);
    float result = (rhs == 0.0f) ? mask_f32() : (lhs / rhs);
    store_f16(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_f16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    float lhs = load_f16(a, idx, data);
    float rhs = load_f16(b, idx, data);
    float result = (rhs == 0.0f) ? mask_f32() : (lhs / rhs);
    store_f16(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    float lhs = load_bf16(a, idx, data);
    float rhs = load_bf16(b, idx, data);
    float result = (rhs == 0.0f) ? mask_f32() : (lhs / rhs);
    store_bf16(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_bf16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    float lhs = load_bf16(a, idx, data);
    float rhs = load_bf16(b, idx, data);
    float result = (rhs == 0.0f) ? mask_f32() : (lhs / rhs);
    store_bf16(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    float lhs = load_f8(a, idx, data);
    float rhs = load_f8(b, idx, data);
    float result = (rhs == 0.0f) ? mask_f32() : (lhs / rhs);
    store_f8(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_f8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    float lhs = load_f8(a, idx, data);
    float rhs = load_f8(b, idx, data);
    float result = (rhs == 0.0f) ? mask_f32() : (lhs / rhs);
    store_f8(a, idx, result, out_buf);
}

int mask_i32() {
    uint64_t bits = (uint64_t(push.mask_i64_hi) << 32u) | uint64_t(push.mask_i64_lo);
    return int(bits);
}

uint mask_u32() {
    uint64_t bits = (uint64_t(push.mask_u64_hi) << 32u) | uint64_t(push.mask_u64_lo);
    return uint(bits);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int lhs = load_i32(a, idx, data);
    int rhs = load_i32(b, idx, data);
    int result = (rhs == 0) ? mask_i32() : (lhs / rhs);
    store_i32(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_i32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int lhs = load_i32(a, idx, data);
    int rhs = load_i32(b, idx, data);
    int result = (rhs == 0) ? mask_i32() : (lhs / rhs);
    store_i32(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_u32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint lhs = load_u32(a, idx, data);
    uint rhs = load_u32(b, idx, data);
    uint result = (rhs == 0u) ? mask_u32() : (lhs / rhs);
    store_u32(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_u32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint lhs = load_u32(a, idx, data);
    uint rhs = load_u32(b, idx, data);
    uint result = (rhs == 0u) ? mask_u32() : (lhs / rhs);
    store_u32(a, idx, result, out_buf);
}

#if HAS_I64
int64_t mask_i64() {
    return int64_t((uint64_t(push.mask_i64_hi) << 32u) | uint64_t(push.mask_i64_lo));
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int64_t lhs = load_i64(a, idx, data);
    int64_t rhs = load_i64(b, idx, data);
    int64_t result = (rhs == 0) ? mask_i64() : (lhs / rhs);
    store_i64(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_i64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int64_t lhs = load_i64(a, idx, data);
    int64_t rhs = load_i64(b, idx, data);
    int64_t result = (rhs == 0) ? mask_i64() : (lhs / rhs);
    store_i64(a, idx, result, out_buf);
}
#endif

#if HAS_U64
uint64_t mask_u64() {
    return (uint64_t(push.mask_u64_hi) << 32u) | uint64_t(push.mask_u64_lo);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_u64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint64_t lhs = load_u64(a, idx, data);
    uint64_t rhs = load_u64(b, idx, data);
    uint64_t result = (rhs == 0u) ? mask_u64() : (lhs / rhs);
    store_u64(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_u64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint64_t lhs = load_u64(a, idx, data);
    uint64_t rhs = load_u64(b, idx, data);
    uint64_t result = (rhs == 0u) ? mask_u64() : (lhs / rhs);
    store_u64(a, idx, result, out_buf);
}
#endif

#if HAS_F64
double mask_f64() {
    uint2 bits = uint2(push.mask_f64_lo, push.mask_f64_hi);
    return asdouble(bits);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    double lhs = load_f64(a, idx, data);
    double rhs = load_f64(b, idx, data);
    double result = (rhs == 0.0) ? mask_f64() : (lhs / rhs);
    store_f64(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_f64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    double lhs = load_f64(a, idx, data);
    double rhs = load_f64(b, idx, data);
    double result = (rhs == 0.0) ? mask_f64() : (lhs / rhs);
    store_f64(a, idx, result, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void div_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int lhs = load_i16(a, idx, data);
    int rhs = load_i16(b, idx, data);
    int result = (rhs == 0) ? mask_i32() : (lhs / rhs);
    store_i16(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_i16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int lhs = load_i16(a, idx, data);
    int rhs = load_i16(b, idx, data);
    int result = (rhs == 0) ? mask_i32() : (lhs / rhs);
    store_i16(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_u16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint lhs = load_u16_val(a, idx, data);
    uint rhs = load_u16_val(b, idx, data);
    uint result = (rhs == 0u) ? mask_u32() : (lhs / rhs);
    store_u16_val(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_u16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint lhs = load_u16_val(a, idx, data);
    uint rhs = load_u16_val(b, idx, data);
    uint result = (rhs == 0u) ? mask_u32() : (lhs / rhs);
    store_u16_val(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int lhs = load_i8(a, idx, data);
    int rhs = load_i8(b, idx, data);
    int result = (rhs == 0) ? mask_i32() : (lhs / rhs);
    store_i8(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_i8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int lhs = load_i8(a, idx, data);
    int rhs = load_i8(b, idx, data);
    int result = (rhs == 0) ? mask_i32() : (lhs / rhs);
    store_i8(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_u8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint lhs = load_u8_val(a, idx, data);
    uint rhs = load_u8_val(b, idx, data);
    uint result = (rhs == 0u) ? mask_u32() : (lhs / rhs);
    store_u8_val(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void div_u8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint lhs = load_u8_val(a, idx, data);
    uint rhs = load_u8_val(b, idx, data);
    uint result = (rhs == 0u) ? mask_u32() : (lhs / rhs);
    store_u8_val(a, idx, result, out_buf);
}
