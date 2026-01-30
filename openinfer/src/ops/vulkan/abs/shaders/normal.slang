#include "../../shaders/common.slang"
#include "../../shaders/float_utils.slang"
#include "common.slang"

#ifndef HAS_F16
#define HAS_F16 0
#endif

#ifndef HAS_I64
#define HAS_I64 0
#endif

#ifndef HAS_F64
#define HAS_F64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { AbsPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float value = load_f32(a, idx, data);
    store_f32(out, idx, abs(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_f32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float value = load_f32(a, idx, data);
    store_f32(a, idx, abs(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_f16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float value = load_f16(a, idx, data);
    store_f16(out, idx, abs(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_f16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float value = load_f16(a, idx, data);
    store_f16(a, idx, abs(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float value = load_bf16(a, idx, data);
    store_bf16(out, idx, abs(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_bf16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float value = load_bf16(a, idx, data);
    store_bf16(a, idx, abs(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float value = load_f8(a, idx, data);
    store_f8(out, idx, abs(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_f8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float value = load_f8(a, idx, data);
    store_f8(a, idx, abs(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int value = load_i32(a, idx, data);
    int result = (value < 0) ? -value : value;
    store_i32(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int value = load_i32(a, idx, data);
    int result = (value < 0) ? -value : value;
    store_i32(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int value = load_i16(a, idx, data);
    int result = (value < 0) ? -value : value;
    store_i16(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int value = load_i16(a, idx, data);
    int result = (value < 0) ? -value : value;
    store_i16(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int value = load_i8(a, idx, data);
    int result = (value < 0) ? -value : value;
    store_i8(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int value = load_i8(a, idx, data);
    int result = (value < 0) ? -value : value;
    store_i8(a, idx, result, out_buf);
}

#if HAS_I64
[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int64_t value = load_i64(a, idx, data);
    int64_t result = (value < 0) ? -value : value;
    store_i64(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int64_t value = load_i64(a, idx, data);
    int64_t result = (value < 0) ? -value : value;
    store_i64(a, idx, result, out_buf);
}
#endif

#if HAS_F64
[numthreads(256, 1, 1)]
[shader("compute")]
void abs_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    double value = load_f64(a, idx, data);
    store_f64(out, idx, abs(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_f64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    double value = load_f64(a, idx, data);
    store_f64(a, idx, abs(value), out_buf);
}
#endif
