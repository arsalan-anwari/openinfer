#include "../../shaders/common.slang"
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
[[vk::push_constant]] cbuffer Push { RemPush push; }

int rem_i32(int lhs, int rhs) {
    if (rhs == 0) return 0;
    return lhs % rhs;
}

uint rem_u32(uint lhs, uint rhs) {
    if (rhs == 0u) return 0u;
    return lhs % rhs;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int lhs = load_i32(a, idx, data);
    int rhs = load_i32(b, idx, data);
    store_i32(out, idx, rem_i32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_i32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int lhs = load_i32(a, idx, data);
    int rhs = load_i32(b, idx, data);
    store_i32(a, idx, rem_i32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_u32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint lhs = load_u32(a, idx, data);
    uint rhs = load_u32(b, idx, data);
    store_u32(out, idx, rem_u32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_u32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint lhs = load_u32(a, idx, data);
    uint rhs = load_u32(b, idx, data);
    store_u32(a, idx, rem_u32(lhs, rhs), out_buf);
}

#if HAS_I64
int64_t rem_i64(int64_t lhs, int64_t rhs) {
    if (rhs == 0) return 0;
    return lhs % rhs;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int64_t lhs = load_i64(a, idx, data);
    int64_t rhs = load_i64(b, idx, data);
    store_i64(out, idx, rem_i64(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_i64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int64_t lhs = load_i64(a, idx, data);
    int64_t rhs = load_i64(b, idx, data);
    store_i64(a, idx, rem_i64(lhs, rhs), out_buf);
}
#endif

#if HAS_U64
uint64_t rem_u64(uint64_t lhs, uint64_t rhs) {
    if (rhs == 0u) return 0u;
    return lhs % rhs;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_u64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint64_t lhs = load_u64(a, idx, data);
    uint64_t rhs = load_u64(b, idx, data);
    store_u64(out, idx, rem_u64(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_u64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint64_t lhs = load_u64(a, idx, data);
    uint64_t rhs = load_u64(b, idx, data);
    store_u64(a, idx, rem_u64(lhs, rhs), out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int lhs = load_i16(a, idx, data);
    int rhs = load_i16(b, idx, data);
    store_i16(out, idx, rem_i32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_i16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int lhs = load_i16(a, idx, data);
    int rhs = load_i16(b, idx, data);
    store_i16(a, idx, rem_i32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_u16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint lhs = load_u16_val(a, idx, data);
    uint rhs = load_u16_val(b, idx, data);
    store_u16_val(out, idx, rem_u32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_u16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint lhs = load_u16_val(a, idx, data);
    uint rhs = load_u16_val(b, idx, data);
    store_u16_val(a, idx, rem_u32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int lhs = load_i8(a, idx, data);
    int rhs = load_i8(b, idx, data);
    store_i8(out, idx, rem_i32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_i8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int lhs = load_i8(a, idx, data);
    int rhs = load_i8(b, idx, data);
    store_i8(a, idx, rem_i32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_u8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint lhs = load_u8_val(a, idx, data);
    uint rhs = load_u8_val(b, idx, data);
    store_u8_val(out, idx, rem_u32(lhs, rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void rem_u8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint lhs = load_u8_val(a, idx, data);
    uint rhs = load_u8_val(b, idx, data);
    store_u8_val(a, idx, rem_u32(lhs, rhs), out_buf);
}
