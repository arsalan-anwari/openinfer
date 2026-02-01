#include "../../shaders/common.slang"
#include "../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { SignPush push; }

int sign_i32(int value) {
    if (value > 0) return 1;
    if (value < 0) return -1;
    return 0;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void sign_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_i(tensor_descs[0], idx, data);
    store_i8(tensor_descs[1], idx, sign_i32(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void sign_i2_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_i(tensor_descs[0], idx, data);
    store_i8(tensor_descs[1], idx, sign_i32(value), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void sign_i1_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int value = load_packed_i(tensor_descs[0], idx, data);
    store_i8(tensor_descs[1], idx, sign_i32(value), out_buf);
}
