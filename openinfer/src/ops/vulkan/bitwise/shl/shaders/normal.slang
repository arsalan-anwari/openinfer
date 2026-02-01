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
[[vk::push_constant]] cbuffer Push { ShlPush push; }

uint shl_bits_u(uint width) {
    return (width == 0u) ? 0u : (push.bits & (width - 1u));
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_i32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int value = load_i32(a, idx, data);
    uint bits = shl_bits_u(32u);
    store_i32(out, idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_i32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int value = load_i32(a, idx, data);
    uint bits = shl_bits_u(32u);
    store_i32(a, idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_u32_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint value = load_u32(a, idx, data);
    uint bits = shl_bits_u(32u);
    store_u32(out, idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_u32_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    uint value = load_u32(a, idx, data);
    uint bits = shl_bits_u(32u);
    store_u32(a, idx, value << bits, out_buf);
}

#if HAS_I64
[numthreads(256, 1, 1)]
[shader("compute")]
void shl_i64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int64_t value = load_i64(a, idx, data);
    uint bits = shl_bits_u(64u);
    store_i64(out, idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_i64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int64_t value = load_i64(a, idx, data);
    uint bits = shl_bits_u(64u);
    store_i64(a, idx, value << bits, out_buf);
}
#endif

#if HAS_U64
[numthreads(256, 1, 1)]
[shader("compute")]
void shl_u64_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint64_t value = load_u64(a, idx, data);
    uint bits = shl_bits_u(64u);
    store_u64(out, idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_u64_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    uint64_t value = load_u64(a, idx, data);
    uint bits = shl_bits_u(64u);
    store_u64(a, idx, value << bits, out_buf);
}
#endif

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_i16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int value = load_i16(a, idx, data);
    uint bits = shl_bits_u(16u);
    store_i16(out, idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_i16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int value = load_i16(a, idx, data);
    uint bits = shl_bits_u(16u);
    store_i16(a, idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_u16_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint value = load_u16_val(a, idx, data);
    uint bits = shl_bits_u(16u);
    store_u16_val(out, idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_u16_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    uint value = load_u16_val(a, idx, data);
    uint bits = shl_bits_u(16u);
    store_u16_val(a, idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_i8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    int value = load_i8(a, idx, data);
    uint bits = shl_bits_u(8u);
    store_i8(out, idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_i8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    int value = load_i8(a, idx, data);
    uint bits = shl_bits_u(8u);
    store_i8(a, idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_u8_normal(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    TensorDesc out = tensor_descs[1];
    uint value = load_u8_val(a, idx, data);
    uint bits = shl_bits_u(8u);
    store_u8_val(out, idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_u8_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    TensorDesc a = tensor_descs[0];
    uint value = load_u8_val(a, idx, data);
    uint bits = shl_bits_u(8u);
    store_u8_val(a, idx, value << bits, out_buf);
}
