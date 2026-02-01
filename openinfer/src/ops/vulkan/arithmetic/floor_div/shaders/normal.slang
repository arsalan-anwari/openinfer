#include "../../../shaders/common.slang"
#include "common.slang"

#ifndef HAS_I64
#define HAS_I64 0
#endif

#ifndef HAS_U64
#define HAS_U64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { FloorDivPush push; }

int mask_i32() {
    uint64_t bits = (uint64_t(push.mask_i64_hi) << 32u) | uint64_t(push.mask_i64_lo);
    return int(bits);
}

uint mask_u32() {
    uint64_t bits = (uint64_t(push.mask_u64_hi) << 32u) | uint64_t(push.mask_u64_lo);
    return uint(bits);
}

int floor_div_i32(int lhs, int rhs) {
    if (rhs == 0) return mask_i32();
    int q = lhs / rhs;
    int r = lhs % rhs;
    if (r < 0) {
        q += (rhs < 0) ? 1 : -1;
    }
    return q;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int lhs = load_i32(a, idx, data);
    int rhs = load_i32(b, idx, data);
    store_i32(out, idx, floor_div_i32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_i32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int lhs = load_i32(a, idx, data);
    int rhs = load_i32(b, idx, data);
    store_i32(a, idx, floor_div_i32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_u32_normal(uint3 tid : SV_DispatchThreadID) {
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
void floor_div_u32_inplace(uint3 tid : SV_DispatchThreadID) {
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

int64_t floor_div_i64(int64_t lhs, int64_t rhs) {
    if (rhs == 0) return mask_i64();
    int64_t q = lhs / rhs;
    int64_t r = lhs % rhs;
    if (r < 0) {
        q += (rhs < 0) ? 1 : -1;
    }
    return q;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int64_t lhs = load_i64(a, idx, data);
    int64_t rhs = load_i64(b, idx, data);
    store_i64(out, idx, floor_div_i64(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_i64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int64_t lhs = load_i64(a, idx, data);
    int64_t rhs = load_i64(b, idx, data);
    store_i64(a, idx, floor_div_i64(lhs, rhs), out_buf);
}
#endif

#if HAS_U64
uint64_t mask_u64() {
    return (uint64_t(push.mask_u64_hi) << 32u) | uint64_t(push.mask_u64_lo);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_u64_normal(uint3 tid : SV_DispatchThreadID) {
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
void floor_div_u64_inplace(uint3 tid : SV_DispatchThreadID) {
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

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int lhs = load_i16(a, idx, data);
    int rhs = load_i16(b, idx, data);
    store_i16(out, idx, floor_div_i32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_i16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int lhs = load_i16(a, idx, data);
    int rhs = load_i16(b, idx, data);
    store_i16(a, idx, floor_div_i32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_u16_normal(uint3 tid : SV_DispatchThreadID) {
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
void floor_div_u16_inplace(uint3 tid : SV_DispatchThreadID) {
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
void floor_div_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int lhs = load_i8(a, idx, data);
    int rhs = load_i8(b, idx, data);
    store_i8(out, idx, floor_div_i32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_i8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int lhs = load_i8(a, idx, data);
    int rhs = load_i8(b, idx, data);
    store_i8(a, idx, floor_div_i32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_u8_normal(uint3 tid : SV_DispatchThreadID) {
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
void floor_div_u8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint lhs = load_u8_val(a, idx, data);
    uint rhs = load_u8_val(b, idx, data);
    uint result = (rhs == 0u) ? mask_u32() : (lhs / rhs);
    store_u8_val(a, idx, result, out_buf);
}
