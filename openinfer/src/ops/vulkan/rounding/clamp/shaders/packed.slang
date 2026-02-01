#include "../../../shaders/common.slang"
#include "../../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { ClampPush push; }

int min_i32() {
    uint64_t bits = (uint64_t(push.min_i64_hi) << 32u) | uint64_t(push.min_i64_lo);
    return int(bits);
}

int max_i32() {
    uint64_t bits = (uint64_t(push.max_i64_hi) << 32u) | uint64_t(push.max_i64_lo);
    return int(bits);
}

uint min_u32() {
    uint64_t bits = (uint64_t(push.min_u64_hi) << 32u) | uint64_t(push.min_u64_lo);
    return uint(bits);
}

uint max_u32() {
    uint64_t bits = (uint64_t(push.max_u64_hi) << 32u) | uint64_t(push.max_u64_lo);
    return uint(bits);
}

int clamp_i_packed(int value, uint bit_width) {
    int min_range = -(1 << (bit_width - 1));
    int max_range = (1 << (bit_width - 1)) - 1;
    int min_val = max(min_i32(), min_range);
    int max_val = min(max_i32(), max_range);
    return min(max(value, min_val), max_val);
}

uint clamp_u_packed(uint value, uint bit_width) {
    uint max_range = (1u << bit_width) - 1u;
    uint min_val = max(min_u32(), 0u);
    uint max_val = min(max_u32(), max_range);
    return min(max(value, min_val), max_val);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    int value = load_packed_i(tensor_descs[0], idx, data);
    int clamped = clamp_i_packed(value, width);
    store_packed(tensor_descs[1], idx, uint(clamped), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    int value = load_packed_i(tensor_descs[0], idx, data);
    int clamped = clamp_i_packed(value, width);
    store_packed(tensor_descs[0], idx, uint(clamped), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    int value = load_packed_i(tensor_descs[0], idx, data);
    int clamped = clamp_i_packed(value, width);
    store_packed(tensor_descs[1], idx, uint(clamped), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    int value = load_packed_i(tensor_descs[0], idx, data);
    int clamped = clamp_i_packed(value, width);
    store_packed(tensor_descs[0], idx, uint(clamped), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    int value = load_packed_i(tensor_descs[0], idx, data);
    int clamped = clamp_i_packed(value, width);
    store_packed(tensor_descs[1], idx, uint(clamped), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_i1_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    int value = load_packed_i(tensor_descs[0], idx, data);
    int clamped = clamp_i_packed(value, width);
    store_packed(tensor_descs[0], idx, uint(clamped), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = load_packed_u(tensor_descs[0], idx, data);
    uint clamped = clamp_u_packed(value, width);
    store_packed(tensor_descs[1], idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = load_packed_u(tensor_descs[0], idx, data);
    uint clamped = clamp_u_packed(value, width);
    store_packed(tensor_descs[0], idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = load_packed_u(tensor_descs[0], idx, data);
    uint clamped = clamp_u_packed(value, width);
    store_packed(tensor_descs[1], idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = load_packed_u(tensor_descs[0], idx, data);
    uint clamped = clamp_u_packed(value, width);
    store_packed(tensor_descs[0], idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = load_packed_u(tensor_descs[0], idx, data);
    uint clamped = clamp_u_packed(value, width);
    store_packed(tensor_descs[1], idx, clamped, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void clamp_u1_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = load_packed_u(tensor_descs[0], idx, data);
    uint clamped = clamp_u_packed(value, width);
    store_packed(tensor_descs[0], idx, clamped, out_buf);
}
