#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { FillPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint value = push.value_u8;
    store_packed(tensor_descs[1], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint value = push.value_u8;
    store_packed(tensor_descs[0], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint value = push.value_u8;
    store_packed(tensor_descs[1], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint value = push.value_u8;
    store_packed(tensor_descs[0], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint value = push.value_u8;
    store_packed(tensor_descs[1], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_i1_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint value = push.value_u8;
    store_packed(tensor_descs[0], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint value = push.value_u8;
    store_packed(tensor_descs[1], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint value = push.value_u8;
    store_packed(tensor_descs[0], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint value = push.value_u8;
    store_packed(tensor_descs[1], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint value = push.value_u8;
    store_packed(tensor_descs[0], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint value = push.value_u8;
    store_packed(tensor_descs[1], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void fill_u1_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint value = push.value_u8;
    store_packed(tensor_descs[0], idx, value, out_buf);
}
