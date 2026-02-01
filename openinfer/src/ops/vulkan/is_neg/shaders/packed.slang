#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { IsNegPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int v = load_packed_i(tensor_descs[0], idx, data);
    store_bool_at(tensor_descs[1], idx, v < 0 ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_i2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int v = load_packed_i(tensor_descs[0], idx, data);
    store_bool_at(tensor_descs[1], idx, v < 0 ? 1u : 0u, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void is_neg_i1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int v = load_packed_i(tensor_descs[0], idx, data);
    store_bool_at(tensor_descs[1], idx, v < 0 ? 1u : 0u, out_buf);
}
