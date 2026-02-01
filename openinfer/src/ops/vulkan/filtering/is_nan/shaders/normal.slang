#include "../../../shaders/common.slang"
#include "common.slang"

#ifndef HAS_F64
#define HAS_F64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { IsNanPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void is_nan_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float v = load_f8(a, idx, data);
    store_bool_at(out, idx, isnan(v) ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_nan_f16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float v = load_f16(a, idx, data);
    store_bool_at(out, idx, isnan(v) ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_nan_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float v = load_bf16(a, idx, data);
    store_bool_at(out, idx, isnan(v) ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_nan_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float v = load_f32(a, idx, data);
    store_bool_at(out, idx, isnan(v) ? 1u : 0u, out_buf);
}

#if HAS_F64
[numthreads(256, 1, 1)]
[shader("compute")]
void is_nan_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    double v = load_f64(a, idx, data);
    store_bool_at(out, idx, isnan(v) ? 1u : 0u, out_buf);
}
#endif
