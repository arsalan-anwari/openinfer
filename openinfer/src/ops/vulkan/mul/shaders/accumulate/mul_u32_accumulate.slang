#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct MulAccBuffersUint64 {
    [[vk::binding(0, 0)]] StructuredBuffer<uint, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint64_t, Std430DataLayout> output0;
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
};

ParameterBlock<MulAccBuffersUint64> gMulAcc_u32_u64;

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_u32_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, gMulAcc_u32_u64.meta0);
        uint b_idx = broadcastOffset(i, gMulAcc_u32_u64.meta1);
        uint64_t a = (uint64_t)gMulAcc_u32_u64.input0[a_idx];
        uint64_t b = (uint64_t)gMulAcc_u32_u64.input1[b_idx];
        gMulAcc_u32_u64.output0[i] = a * b;
    }
}
