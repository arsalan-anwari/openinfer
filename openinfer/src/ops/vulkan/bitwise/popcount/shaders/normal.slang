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
[[vk::push_constant]] cbuffer Push { PopcountPush push; }

uint popcount_u64(uint64_t value) {
    uint lo = (uint)(value & 0xFFFFFFFFu);
    uint hi = (uint)(value >> 32);
    return countbits(lo) + countbits(hi);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void popcount_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint value = (uint)load_i32(a, idx, data);
    store_u8_val(out, idx, countbits(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void popcount_u32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint value = load_u32(a, idx, data);
    store_u8_val(out, idx, countbits(value), out_buf);
}

#if HAS_I64
[numthreads(256, 1, 1)]
[shader("compute")]
void popcount_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint64_t value = (uint64_t)load_i64(a, idx, data);
    store_u8_val(out, idx, popcount_u64(value), out_buf);
}
#endif

#if HAS_U64
[numthreads(256, 1, 1)]
[shader("compute")]
void popcount_u64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint64_t value = load_u64(a, idx, data);
    store_u8_val(out, idx, popcount_u64(value), out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void popcount_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint value = (uint)load_i16(a, idx, data);
    store_u8_val(out, idx, countbits(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void popcount_u16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint value = load_u16_val(a, idx, data);
    store_u8_val(out, idx, countbits(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void popcount_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint value = load_u8_val(a, idx, data);
    store_u8_val(out, idx, countbits(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void popcount_u8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint value = load_u8_val(a, idx, data);
    store_u8_val(out, idx, countbits(value), out_buf);
}
