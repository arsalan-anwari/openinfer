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
[[vk::push_constant]] cbuffer Push { AndPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int result = load_i32(a, idx, data) & load_i32(b, idx, data);
    store_i32(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int result = load_i32(a, idx, data) & load_i32(b, idx, data);
    store_i32(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint result = load_u32(a, idx, data) & load_u32(b, idx, data);
    store_u32(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint result = load_u32(a, idx, data) & load_u32(b, idx, data);
    store_u32(a, idx, result, out_buf);
}

#if HAS_I64
[numthreads(256, 1, 1)]
[shader("compute")]
void and_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int64_t result = load_i64(a, idx, data) & load_i64(b, idx, data);
    store_i64(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int64_t result = load_i64(a, idx, data) & load_i64(b, idx, data);
    store_i64(a, idx, result, out_buf);
}
#endif

#if HAS_U64
[numthreads(256, 1, 1)]
[shader("compute")]
void and_u64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint64_t result = load_u64(a, idx, data) & load_u64(b, idx, data);
    store_u64(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint64_t result = load_u64(a, idx, data) & load_u64(b, idx, data);
    store_u64(a, idx, result, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int result = load_i16(a, idx, data) & load_i16(b, idx, data);
    store_i16(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int result = load_i16(a, idx, data) & load_i16(b, idx, data);
    store_i16(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint result = load_u16_val(a, idx, data) & load_u16_val(b, idx, data);
    store_u16_val(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint result = load_u16_val(a, idx, data) & load_u16_val(b, idx, data);
    store_u16_val(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    int result = load_i8(a, idx, data) & load_i8(b, idx, data);
    store_i8(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    int result = load_i8(a, idx, data) & load_i8(b, idx, data);
    store_i8(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint result = load_u8_val(a, idx, data) & load_u8_val(b, idx, data);
    store_u8_val(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint result = load_u8_val(a, idx, data) & load_u8_val(b, idx, data);
    store_u8_val(a, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_bool_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    TensorDesc out = tensor_descs[2];
    uint result = load_bool_at(a, idx, data) & load_bool_at(b, idx, data);
    store_bool_at(out, idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_bool_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc b = tensor_descs[1];
    uint result = load_bool_at(a, idx, data) & load_bool_at(b, idx, data);
    store_bool_at(a, idx, result, out_buf);
}
