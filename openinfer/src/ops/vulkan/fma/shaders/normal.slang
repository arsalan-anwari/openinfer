#include "../../shaders/common.slang"
#include "common.slang"

#ifndef HAS_F64
#define HAS_F64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { FmaPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void fma_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc c = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    float value = load_f8(a, idx, data) * load_f8(b, idx, data) + load_f8(c, idx, data);
    store_f8(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fma_f8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc c = tensor_descs[2];
    float value = load_f8(a, idx, data) * load_f8(b, idx, data) + load_f8(c, idx, data);
    store_f8(a, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fma_f16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc c = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    float value = load_f16(a, idx, data) * load_f16(b, idx, data) + load_f16(c, idx, data);
    store_f16(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fma_f16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc c = tensor_descs[2];
    float value = load_f16(a, idx, data) * load_f16(b, idx, data) + load_f16(c, idx, data);
    store_f16(a, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fma_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc c = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    float value = load_bf16(a, idx, data) * load_bf16(b, idx, data) + load_bf16(c, idx, data);
    store_bf16(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fma_bf16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc c = tensor_descs[2];
    float value = load_bf16(a, idx, data) * load_bf16(b, idx, data) + load_bf16(c, idx, data);
    store_bf16(a, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fma_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc c = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    float value = load_f32(a, idx, data) * load_f32(b, idx, data) + load_f32(c, idx, data);
    store_f32(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fma_f32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc c = tensor_descs[2];
    float value = load_f32(a, idx, data) * load_f32(b, idx, data) + load_f32(c, idx, data);
    store_f32(a, idx, value, out_buf);
}

#if HAS_F64
[numthreads(256, 1, 1)]
[shader("compute")]
void fma_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc c = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    double value = load_f64(a, idx, data) * load_f64(b, idx, data) + load_f64(c, idx, data);
    store_f64(out, idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fma_f64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc c = tensor_descs[2];
    double value = load_f64(a, idx, data) * load_f64(b, idx, data) + load_f64(c, idx, data);
    store_f64(a, idx, value, out_buf);
}
#endif
