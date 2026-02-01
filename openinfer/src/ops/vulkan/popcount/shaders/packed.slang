#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { PopcountPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void popcount_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = uint(load_packed_i(tensor_descs[0], idx, data));
    uint mask = mask_n(width);
    store_u8_val(tensor_descs[1], idx, countbits(value & mask), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void popcount_i2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = uint(load_packed_i(tensor_descs[0], idx, data));
    uint mask = mask_n(width);
    store_u8_val(tensor_descs[1], idx, countbits(value & mask), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void popcount_u4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = load_packed_u(tensor_descs[0], idx, data);
    uint mask = mask_n(width);
    store_u8_val(tensor_descs[1], idx, countbits(value & mask), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void popcount_u2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint width = tensor_descs[0].elem_bits;
    uint value = load_packed_u(tensor_descs[0], idx, data);
    uint mask = mask_n(width);
    store_u8_val(tensor_descs[1], idx, countbits(value & mask), out_buf);
}
