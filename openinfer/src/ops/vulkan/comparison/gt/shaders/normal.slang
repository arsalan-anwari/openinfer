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
[[vk::push_constant]] cbuffer Push { GtPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void gt_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    float lhs = load_f8(a, idx, data);
    float rhs = load_f8(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void gt_f16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    float lhs = load_f16(a, idx, data);
    float rhs = load_f16(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void gt_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    float lhs = load_bf16(a, idx, data);
    float rhs = load_bf16(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void gt_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    float lhs = load_f32(a, idx, data);
    float rhs = load_f32(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}

#if HAS_F64
[numthreads(256, 1, 1)]
[shader("compute")]
void gt_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    double lhs = load_f64(a, idx, data);
    double rhs = load_f64(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void gt_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int lhs = load_i32(a, idx, data);
    int rhs = load_i32(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}

#if HAS_I64
[numthreads(256, 1, 1)]
[shader("compute")]
void gt_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int64_t lhs = load_i64(a, idx, data);
    int64_t rhs = load_i64(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void gt_u32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint lhs = load_u32(a, idx, data);
    uint rhs = load_u32(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}

#if HAS_U64
[numthreads(256, 1, 1)]
[shader("compute")]
void gt_u64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint64_t lhs = load_u64(a, idx, data);
    uint64_t rhs = load_u64(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void gt_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int lhs = load_i16(a, idx, data);
    int rhs = load_i16(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void gt_u16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint lhs = load_u16_val(a, idx, data);
    uint rhs = load_u16_val(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void gt_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int lhs = load_i8(a, idx, data);
    int rhs = load_i8(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void gt_u8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint lhs = load_u8_val(a, idx, data);
    uint rhs = load_u8_val(b, idx, data);
    store_bool_at(out, idx, lhs > rhs ? 1u : 0u, out_buf);
}
