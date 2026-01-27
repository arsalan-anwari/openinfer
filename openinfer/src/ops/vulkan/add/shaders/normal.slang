#include "../../shaders/common.slang"
#include "../../shaders/float_utils.slang"
#include "common.slang"

#ifndef HAS_F16
#define HAS_F16 0
#endif

#ifndef HAS_I64
#define HAS_I64 0
#endif

#ifndef HAS_U64
#define HAS_U64 0
#endif

#ifndef HAS_F64
#define HAS_F64 0
#endif

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { AddPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void add_f32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    float result = load_f32(a, idx, data) + load_f32(b, idx, data);
    store_f32(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_f32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    float result = load_f32(a, idx, data) + load_f32(b, idx, data);
    store_f32(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
#if HAS_F16
void add_f16_normal_native(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    half ah = (half)load_f16(a, idx, data);
    half bh = (half)load_f16(b, idx, data);
    half ch = ah + bh;
    store_f16(out, idx, (float)ch, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_f16_inplace_native(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    half ah = (half)load_f16(a, idx, data);
    half bh = (half)load_f16(b, idx, data);
    half ch = ah + bh;
    store_f16(a, idx, (float)ch, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void add_f16_normal_simulated(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    float result = load_f16(a, idx, data) + load_f16(b, idx, data);
    store_f16(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_f16_inplace_simulated(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    float result = load_f16(a, idx, data) + load_f16(b, idx, data);
    store_f16(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_bf16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    float result = load_bf16(a, idx, data) + load_bf16(b, idx, data);
    store_bf16(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_bf16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    float result = load_bf16(a, idx, data) + load_bf16(b, idx, data);
    store_bf16(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_f8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    float result = load_f8(a, idx, data) + load_f8(b, idx, data);
    store_f8(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_f8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    float result = load_f8(a, idx, data) + load_f8(b, idx, data);
    store_f8(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int result = load_i32(a, idx, data) + load_i32(b, idx, data);
    store_i32(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int result = load_i32(a, idx, data) + load_i32(b, idx, data);
    store_i32(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint result = load_u32(a, idx, data) + load_u32(b, idx, data);
    store_u32(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint result = load_u32(a, idx, data) + load_u32(b, idx, data);
    store_u32(a, idx, result, out_buf);
}

#if HAS_I64
[numthreads(256, 1, 1)]
[shader("compute")]
void add_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int64_t result = load_i64(a, idx, data) + load_i64(b, idx, data);
    store_i64(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int64_t result = load_i64(a, idx, data) + load_i64(b, idx, data);
    store_i64(a, idx, result, out_buf);
}
#endif

#if HAS_U64
[numthreads(256, 1, 1)]
[shader("compute")]
void add_u64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint64_t result = load_u64(a, idx, data) + load_u64(b, idx, data);
    store_u64(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint64_t result = load_u64(a, idx, data) + load_u64(b, idx, data);
    store_u64(a, idx, result, out_buf);
}
#endif

#if HAS_F64
[numthreads(256, 1, 1)]
[shader("compute")]
void add_f64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    double result = load_f64(a, idx, data) + load_f64(b, idx, data);
    store_f64(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_f64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    double result = load_f64(a, idx, data) + load_f64(b, idx, data);
    store_f64(a, idx, result, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int result = load_i16(a, idx, data) + load_i16(b, idx, data);
    store_i16(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int result = load_i16(a, idx, data) + load_i16(b, idx, data);
    store_i16(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint result = load_u16_val(a, idx, data) + load_u16_val(b, idx, data);
    store_u16_val(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint result = load_u16_val(a, idx, data) + load_u16_val(b, idx, data);
    store_u16_val(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int result = load_i8(a, idx, data) + load_i8(b, idx, data);
    store_i8(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_i8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int result = load_i8(a, idx, data) + load_i8(b, idx, data);
    store_i8(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint result = load_u8_val(a, idx, data) + load_u8_val(b, idx, data);
    store_u8_val(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_u8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint result = load_u8_val(a, idx, data) + load_u8_val(b, idx, data);
    store_u8_val(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_bool_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint result = (load_bool_at(a, idx, data) + load_bool_at(b, idx, data)) != 0u;
    store_bool_at(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_bool_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint result = (load_bool_at(a, idx, data) + load_bool_at(b, idx, data)) != 0u;
    store_bool_at(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_bitset_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint result = load_bitset_at(a, idx, data) + load_bitset_at(b, idx, data);
    store_bitset_at(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void add_bitset_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint result = load_bitset_at(a, idx, data) + load_bitset_at(b, idx, data);
    store_bitset_at(a, idx, result, out_buf);
}
