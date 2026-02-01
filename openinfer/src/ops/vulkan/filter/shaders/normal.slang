#include "../../shaders/common.slang"
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
[[vk::push_constant]] cbuffer Push { FilterPush push; }

uint cond_f32(float v) { return v != 0.0f ? 1u : 0u; }
uint cond_f64(double v) { return v != 0.0 ? 1u : 0u; }

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = cond_f32(load_f8(cond, idx, data));
    uint offset_a = linear_offset(a, idx);
    uint offset_b = linear_offset(b, idx);
    uint bits = choose_a != 0u ? load_u8(data, offset_a) : load_u8(data, offset_b);
    store_u8(out_buf, linear_offset(out, idx), bits);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_f16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = cond_f32(load_f16(cond, idx, data));
    uint offset_a = linear_offset(a, idx);
    uint offset_b = linear_offset(b, idx);
    uint bits = choose_a != 0u ? load_u16(data, offset_a) : load_u16(data, offset_b);
    store_u16(out_buf, linear_offset(out, idx), bits);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = cond_f32(load_bf16(cond, idx, data));
    uint offset_a = linear_offset(a, idx);
    uint offset_b = linear_offset(b, idx);
    uint bits = choose_a != 0u ? load_u16(data, offset_a) : load_u16(data, offset_b);
    store_u16(out_buf, linear_offset(out, idx), bits);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = cond_f32(load_f32(cond, idx, data));
    float result = choose_a != 0u ? load_f32(a, idx, data) : load_f32(b, idx, data);
    store_f32(out, idx, result, out_buf);
}

#if HAS_F64
[numthreads(256, 1, 1)]
[shader("compute")]
void filter_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = cond_f64(load_f64(cond, idx, data));
    double result = choose_a != 0u ? load_f64(a, idx, data) : load_f64(b, idx, data);
    store_f64(out, idx, result, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = load_i32(cond, idx, data) != 0 ? 1u : 0u;
    int result = choose_a != 0u ? load_i32(a, idx, data) : load_i32(b, idx, data);
    store_i32(out, idx, result, out_buf);
}

#if HAS_I64
[numthreads(256, 1, 1)]
[shader("compute")]
void filter_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = load_i64(cond, idx, data) != 0 ? 1u : 0u;
    int64_t result = choose_a != 0u ? load_i64(a, idx, data) : load_i64(b, idx, data);
    store_i64(out, idx, result, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_u32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = load_u32(cond, idx, data) != 0u ? 1u : 0u;
    uint result = choose_a != 0u ? load_u32(a, idx, data) : load_u32(b, idx, data);
    store_u32(out, idx, result, out_buf);
}

#if HAS_U64
[numthreads(256, 1, 1)]
[shader("compute")]
void filter_u64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = load_u64(cond, idx, data) != 0u ? 1u : 0u;
    uint64_t result = choose_a != 0u ? load_u64(a, idx, data) : load_u64(b, idx, data);
    store_u64(out, idx, result, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = load_i16(cond, idx, data) != 0 ? 1u : 0u;
    int result = choose_a != 0u ? load_i16(a, idx, data) : load_i16(b, idx, data);
    store_i16(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_u16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = load_u16_val(cond, idx, data) != 0u ? 1u : 0u;
    uint result = choose_a != 0u ? load_u16_val(a, idx, data) : load_u16_val(b, idx, data);
    store_u16_val(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = load_i8(cond, idx, data) != 0 ? 1u : 0u;
    int result = choose_a != 0u ? load_i8(a, idx, data) : load_i8(b, idx, data);
    store_i8(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_u8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = load_u8_val(cond, idx, data) != 0u ? 1u : 0u;
    uint result = choose_a != 0u ? load_u8_val(a, idx, data) : load_u8_val(b, idx, data);
    store_u8_val(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_bool_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc cond = tensor_descs[2];
    TensorDesc out = tensor_descs[3];
    uint choose_a = load_bool_at(cond, idx, data);
    uint result = choose_a != 0u ? load_bool_at(a, idx, data) : load_bool_at(b, idx, data);
    store_bool_at(out, idx, result, out_buf);
}
