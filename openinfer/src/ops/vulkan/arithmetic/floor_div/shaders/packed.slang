#include "../../../shaders/common.slang"
#include "../../../shaders/packed_utils.slang"
#include "common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TensorDesc> tensor_descs;
[[vk::binding(1, 0)]] ByteAddressBuffer data;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] cbuffer Push { FloorDivPush push; }

int mask_i32() {
    uint64_t bits = (uint64_t(push.mask_i64_hi) << 32u) | uint64_t(push.mask_i64_lo);
    return int(bits);
}

uint mask_u32() {
    uint64_t bits = (uint64_t(push.mask_u64_hi) << 32u) | uint64_t(push.mask_u64_lo);
    return uint(bits);
}

int floor_div_i32(int lhs, int rhs) {
    if (rhs == 0) return mask_i32();
    int q = lhs / rhs;
    int r = lhs % rhs;
    if (r < 0) {
        q += (rhs < 0) ? 1 : -1;
    }
    return q;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_i4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int lhs = load_packed_i(tensor_descs[0], idx, data);
    int rhs = load_packed_i(tensor_descs[1], idx, data);
    int result = floor_div_i32(lhs, rhs);
    store_packed(tensor_descs[2], idx, uint(result), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_i4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    int lhs = load_packed_i(tensor_descs[0], idx, data);
    int rhs = load_packed_i(tensor_descs[1], idx, data);
    int result = floor_div_i32(lhs, rhs);
    store_packed(tensor_descs[0], idx, uint(result), out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_u4_packed(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    uint result = (rhs == 0u) ? mask_u32() : (lhs / rhs);
    store_packed(tensor_descs[2], idx, result, out_buf);
}

[numthreads(256, 1, 1)]
[shader("compute")]
void floor_div_u4_packed_inplace(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= push.len) return;
    uint lhs = load_packed_u(tensor_descs[0], idx, data);
    uint rhs = load_packed_u(tensor_descs[1], idx, data);
    uint result = (rhs == 0u) ? mask_u32() : (lhs / rhs);
    store_packed(tensor_descs[0], idx, result, out_buf);
}
