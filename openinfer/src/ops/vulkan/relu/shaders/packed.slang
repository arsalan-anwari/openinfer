#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { ReluPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int alpha = push.alpha_i32;
    int clamp_max = push.clamp_max_i32;
    int x = load_packed_i(tensor_descs[0], idx, data);
    int y = (x >= 0) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_packed(tensor_descs[1], idx, uint(y), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void relu_i4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int alpha = push.alpha_i32;
    int clamp_max = push.clamp_max_i32;
    int x = load_packed_i(tensor_descs[0], idx, data);
    int y = (x >= 0) ? x : (x * alpha);
    if (y > clamp_max) y = clamp_max;
    store_packed(tensor_descs[0], idx, uint(y), out_buf);
}
