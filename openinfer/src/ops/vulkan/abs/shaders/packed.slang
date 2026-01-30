#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { AbsPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_i(tensor_descs[0], idx, data);
    int result = (value < 0) ? -value : value;
    store_packed(tensor_descs[1], idx, uint(result), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_i(tensor_descs[0], idx, data);
    int result = (value < 0) ? -value : value;
    store_packed(tensor_descs[0], idx, uint(result), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_i(tensor_descs[0], idx, data);
    int result = (value < 0) ? -value : value;
    store_packed(tensor_descs[1], idx, uint(result), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_i(tensor_descs[0], idx, data);
    int result = (value < 0) ? -value : value;
    store_packed(tensor_descs[0], idx, uint(result), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_i(tensor_descs[0], idx, data);
    int result = (value < 0) ? -value : value;
    store_packed(tensor_descs[1], idx, uint(result), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void abs_i1_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_i(tensor_descs[0], idx, data);
    int result = (value < 0) ? -value : value;
    store_packed(tensor_descs[0], idx, uint(result), out_buf);
}
