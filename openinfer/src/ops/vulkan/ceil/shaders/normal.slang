#include "../../shaders/common.slang"
#include "../../shaders/float_utils.slang"
#include "common.slang"

#ifndef HAS_F16
#define HAS_F16 0
#endif

#ifndef HAS_F64
#define HAS_F64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { CeilPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void ceil_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float value = load_f32(a, idx, data);
    store_f32(out, idx, ceil(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void ceil_f32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float value = load_f32(a, idx, data);
    store_f32(a, idx, ceil(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void ceil_f16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float value = load_f16(a, idx, data);
    store_f16(out, idx, ceil(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void ceil_f16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float value = load_f16(a, idx, data);
    store_f16(a, idx, ceil(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void ceil_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float value = load_bf16(a, idx, data);
    store_bf16(out, idx, ceil(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void ceil_bf16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float value = load_bf16(a, idx, data);
    store_bf16(a, idx, ceil(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void ceil_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float value = load_f8(a, idx, data);
    store_f8(out, idx, ceil(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void ceil_f8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float value = load_f8(a, idx, data);
    store_f8(a, idx, ceil(value), out_buf);
}

#if HAS_F64
[numthreads(256, 1, 1)]
[shader("compute")]
void ceil_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    double value = load_f64(a, idx, data);
    store_f64(out, idx, ceil(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void ceil_f64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    double value = load_f64(a, idx, data);
    store_f64(a, idx, ceil(value), out_buf);
}
#endif
