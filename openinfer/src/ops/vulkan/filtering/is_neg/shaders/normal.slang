#include "../../../shaders/common.slang"
#include "common.slang"

#ifndef HAS_F64
#define HAS_F64 0
#endif

#ifndef HAS_I64
#define HAS_I64 0
#endif

#ifndef HAS_U64
#define HAS_U64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { IsNegPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float v = load_f8(a, idx, data);
    uint sign = asuint(v) >> 31u;
    store_bool_at(out, idx, sign, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_f16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float v = load_f16(a, idx, data);
    uint sign = asuint(v) >> 31u;
    store_bool_at(out, idx, sign, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float v = load_bf16(a, idx, data);
    uint sign = asuint(v) >> 31u;
    store_bool_at(out, idx, sign, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float v = load_f32(a, idx, data);
    uint sign = asuint(v) >> 31u;
    store_bool_at(out, idx, sign, out_buf);
}

#if HAS_F64
[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    double v = load_f64(a, idx, data);
    uint2 bits = asuint(v);
    uint sign = bits.y >> 31u;
    store_bool_at(out, idx, sign, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int v = load_i32(a, idx, data);
    store_bool_at(out, idx, v < 0 ? 1u : 0u, out_buf);
}

#if HAS_I64
[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int64_t v = load_i64(a, idx, data);
    store_bool_at(out, idx, v < 0 ? 1u : 0u, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_u32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    store_bool_at(out, idx, 0u, out_buf);
}

#if HAS_U64
[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_u64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    store_bool_at(out, idx, 0u, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int v = load_i16(a, idx, data);
    store_bool_at(out, idx, v < 0 ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_u16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    store_bool_at(out, idx, 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int v = load_i8(a, idx, data);
    store_bool_at(out, idx, v < 0 ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_u8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc out = tensor_descs[1];
    store_bool_at(out, idx, 0u, out_buf);
}
