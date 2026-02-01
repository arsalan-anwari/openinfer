#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { FilterPush push; }

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint cond = uint(load_packed_i(tensor_descs[2], idx, data));
    uint value = cond != 0u
        ? uint(load_packed_i(tensor_descs[0], idx, data))
        : uint(load_packed_i(tensor_descs[1], idx, data));
    store_packed(tensor_descs[3], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_i2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint cond = uint(load_packed_i(tensor_descs[2], idx, data));
    uint value = cond != 0u
        ? uint(load_packed_i(tensor_descs[0], idx, data))
        : uint(load_packed_i(tensor_descs[1], idx, data));
    store_packed(tensor_descs[3], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_i1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint cond = uint(load_packed_i(tensor_descs[2], idx, data));
    uint value = cond != 0u
        ? uint(load_packed_i(tensor_descs[0], idx, data))
        : uint(load_packed_i(tensor_descs[1], idx, data));
    store_packed(tensor_descs[3], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_u4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint cond = load_packed_u(tensor_descs[2], idx, data);
    uint value = cond != 0u
        ? load_packed_u(tensor_descs[0], idx, data)
        : load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[3], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_u2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint cond = load_packed_u(tensor_descs[2], idx, data);
    uint value = cond != 0u
        ? load_packed_u(tensor_descs[0], idx, data)
        : load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[3], idx, value, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void filter_u1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint cond = load_packed_u(tensor_descs[2], idx, data);
    uint value = cond != 0u
        ? load_packed_u(tensor_descs[0], idx, data)
        : load_packed_u(tensor_descs[1], idx, data);
    store_packed(tensor_descs[3], idx, value, out_buf);
}
