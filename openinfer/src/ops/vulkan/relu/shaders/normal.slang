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
[[vk::push_constant]] cbuffer Push { ReluPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float alpha = push.alpha_f32;
    float clamp_max = push.clamp_max_f32;
    float x = load_f32(a, idx, data);
    float y = (x >= 0.0f) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_f32(out, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_f32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float alpha = push.alpha_f32;
    float clamp_max = push.clamp_max_f32;
    float x = load_f32(a, idx, data);
    float y = (x >= 0.0f) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_f32(a, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_f16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float alpha = push.alpha_f32;
    float clamp_max = push.clamp_max_f32;
    float x = load_f16(a, idx, data);
    float y = (x >= 0.0f) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_f16(out, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_f16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float alpha = push.alpha_f32;
    float clamp_max = push.clamp_max_f32;
    float x = load_f16(a, idx, data);
    float y = (x >= 0.0f) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_f16(a, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float alpha = push.alpha_f32;
    float clamp_max = push.clamp_max_f32;
    float x = load_bf16(a, idx, data);
    float y = (x >= 0.0f) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_bf16(out, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_bf16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float alpha = push.alpha_f32;
    float clamp_max = push.clamp_max_f32;
    float x = load_bf16(a, idx, data);
    float y = (x >= 0.0f) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_bf16(a, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    float alpha = push.alpha_f32;
    float clamp_max = push.clamp_max_f32;
    float x = load_f8(a, idx, data);
    float y = (x >= 0.0f) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_f8(out, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_f8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    float alpha = push.alpha_f32;
    float clamp_max = push.clamp_max_f32;
    float x = load_f8(a, idx, data);
    float y = (x >= 0.0f) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_f8(a, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int alpha = push.alpha_i32;
    int clamp_max = push.clamp_max_i32;
    int x = load_i8(a, idx, data);
    int y = (x >= 0) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_i8(out, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_i8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int alpha = push.alpha_i32;
    int clamp_max = push.clamp_max_i32;
    int x = load_i8(a, idx, data);
    int y = (x >= 0) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_i8(a, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int alpha = push.alpha_i32;
    int clamp_max = push.clamp_max_i32;
    int x = load_i16(a, idx, data);
    int y = (x >= 0) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_i16(out, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_i16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int alpha = push.alpha_i32;
    int clamp_max = push.clamp_max_i32;
    int x = load_i16(a, idx, data);
    int y = (x >= 0) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_i16(a, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int alpha = push.alpha_i32;
    int clamp_max = push.clamp_max_i32;
    int x = load_i32(a, idx, data);
    int y = (x >= 0) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_i32(out, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_i32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int alpha = push.alpha_i32;
    int clamp_max = push.clamp_max_i32;
    int x = load_i32(a, idx, data);
    int y = (x >= 0) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_i32(a, idx, y, out_buf);
}

#if HAS_I64
int64_t relu_alpha_i64() {
    uint64_t bits = (uint64_t(push.alpha_i64_hi) << 32) | uint64_t(push.alpha_i64_lo);
    return int64_t(bits);
}

int64_t relu_clamp_i64() {
    uint64_t bits = (uint64_t(push.clamp_i64_hi) << 32) | uint64_t(push.clamp_i64_lo);
    return int64_t(bits);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int64_t alpha = relu_alpha_i64();
    int64_t clamp_max = relu_clamp_i64();
    int64_t x = load_i64(a, idx, data);
    int64_t y = (x >= 0) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_i64(out, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_i64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int64_t alpha = relu_alpha_i64();
    int64_t clamp_max = relu_clamp_i64();
    int64_t x = load_i64(a, idx, data);
    int64_t y = (x >= 0) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_i64(a, idx, y, out_buf);
}
#endif

#if HAS_F64
double relu_alpha_f64() {
    uint2 bits = uint2(push.alpha_f64_lo, push.alpha_f64_hi);
    return asdouble(bits);
}

double relu_clamp_f64() {
    uint2 bits = uint2(push.clamp_f64_lo, push.clamp_f64_hi);
    return asdouble(bits);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    double alpha = relu_alpha_f64();
    double clamp_max = relu_clamp_f64();
    double x = load_f64(a, idx, data);
    double y = (x >= 0.0) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_f64(out, idx, y, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_f64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    double alpha = relu_alpha_f64();
    double clamp_max = relu_clamp_f64();
    double x = load_f64(a, idx, data);
    double y = (x >= 0.0) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_f64(a, idx, y, out_buf);
}
#endif
