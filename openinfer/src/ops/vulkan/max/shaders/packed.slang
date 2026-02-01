#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { MaxPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void max_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int lhs = load_packed_i(tensor_descs[0], idx, data);
    int rhs = load_packed_i(tensor_descs[1], idx, data);
    store_packed(tensor_descs[2], idx, uint((lhs > rhs) ? lhs : rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void max_i4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int lhs = load_packed_i(tensor_descs[0], idx, data);
    int rhs = load_packed_i(tensor_descs[1], idx, data);
    store_packed(tensor_descs[0], idx, uint((lhs > rhs) ? lhs : rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void max_i2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int lhs = load_packed_i(tensor_descs[0], idx, data);
    int rhs = load_packed_i(tensor_descs[1], idx, data);
    store_packed(tensor_descs[2], idx, uint((lhs > rhs) ? lhs : rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void max_i2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int lhs = load_packed_i(tensor_descs[0], idx, data);
    int rhs = load_packed_i(tensor_descs[1], idx, data);
    store_packed(tensor_descs[0], idx, uint((lhs > rhs) ? lhs : rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void max_i1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int lhs = load_packed_i(tensor_descs[0], idx, data);
    int rhs = load_packed_i(tensor_descs[1], idx, data);
    store_packed(tensor_descs[2], idx, uint((lhs > rhs) ? lhs : rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void max_i1_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int lhs = load_packed_i(tensor_descs[0], idx, data);
    int rhs = load_packed_i(tensor_descs[1], idx, data);
    store_packed(tensor_descs[0], idx, uint((lhs > rhs) ? lhs : rhs), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void max_u4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[2], idx, (lhs > rhs) ? lhs : rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void max_u4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[0], idx, (lhs > rhs) ? lhs : rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void max_u2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[2], idx, (lhs > rhs) ? lhs : rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void max_u2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[0], idx, (lhs > rhs) ? lhs : rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void max_u1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[2], idx, (lhs > rhs) ? lhs : rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void max_u1_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[0], idx, (lhs > rhs) ? lhs : rhs, out_buf);
}
