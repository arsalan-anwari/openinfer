#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { AndPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = uint(load_packed_i(tensor_descs[0], idx, data));
    uint rhs = uint(load_packed_i(tensor_descs[1], idx, data));
    store_packed(tensor_descs[2], idx, lhs & rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = uint(load_packed_i(tensor_descs[0], idx, data));
    uint rhs = uint(load_packed_i(tensor_descs[1], idx, data));
    store_packed(tensor_descs[0], idx, lhs & rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = uint(load_packed_i(tensor_descs[0], idx, data));
    uint rhs = uint(load_packed_i(tensor_descs[1], idx, data));
    store_packed(tensor_descs[2], idx, lhs & rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = uint(load_packed_i(tensor_descs[0], idx, data));
    uint rhs = uint(load_packed_i(tensor_descs[1], idx, data));
    store_packed(tensor_descs[0], idx, lhs & rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = uint(load_packed_i(tensor_descs[0], idx, data));
    uint rhs = uint(load_packed_i(tensor_descs[1], idx, data));
    store_packed(tensor_descs[2], idx, lhs & rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_i1_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = uint(load_packed_i(tensor_descs[0], idx, data));
    uint rhs = uint(load_packed_i(tensor_descs[1], idx, data));
    store_packed(tensor_descs[0], idx, lhs & rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[2], idx, lhs & rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[0], idx, lhs & rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[2], idx, lhs & rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u2_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[0], idx, lhs & rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[2], idx, lhs & rhs, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void and_u1_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[0], idx, lhs & rhs, out_buf);
}
