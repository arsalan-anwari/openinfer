#include "../../../shaders/common.slang"
#include "../../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { ShlPush push; }

uint shl_bits_u(uint width) {
    return (width == 0u) ? 0u : (push.bits & (width - 1u));
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    int value = load_packed_i(tensor_descs[0], idx, data);
    uint bits = shl_bits_u(width);
    store_packed(tensor_descs[1], idx, uint(value << bits), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_i4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    int value = load_packed_i(tensor_descs[0], idx, data);
    uint bits = shl_bits_u(width);
    store_packed(tensor_descs[0], idx, uint(value << bits), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_i2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    int value = load_packed_i(tensor_descs[0], idx, data);
    uint bits = shl_bits_u(width);
    store_packed(tensor_descs[1], idx, uint(value << bits), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_i2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    int value = load_packed_i(tensor_descs[0], idx, data);
    uint bits = shl_bits_u(width);
    store_packed(tensor_descs[0], idx, uint(value << bits), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_u4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = load_packed_u(tensor_descs[0], idx, data);
    uint bits = shl_bits_u(width);
    store_packed(tensor_descs[1], idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_u4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = load_packed_u(tensor_descs[0], idx, data);
    uint bits = shl_bits_u(width);
    store_packed(tensor_descs[0], idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_u2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = load_packed_u(tensor_descs[0], idx, data);
    uint bits = shl_bits_u(width);
    store_packed(tensor_descs[1], idx, value << bits, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void shl_u2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = load_packed_u(tensor_descs[0], idx, data);
    uint bits = shl_bits_u(width);
    store_packed(tensor_descs[0], idx, value << bits, out_buf);
}
