#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct MulAccBuffersUint {
    [[vk::binding(0, 0)]] StructuredBuffer<uint, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
#ifdef OI_BROADCAST

    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

struct MulAccBuffersUint64 {
    [[vk::binding(0, 0)]] StructuredBuffer<uint, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint64_t, Std430DataLayout> output0;
#ifdef OI_BROADCAST

    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

ParameterBlock<MulAccBuffersUint> gMulAcc_u8_u16;
ParameterBlock<MulAccBuffersUint> gMulAcc_u8_u32;
ParameterBlock<MulAccBuffersUint64> gMulAcc_u8_u64;

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_u8_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMulAcc_u8_u16.meta0);
        b_idx = broadcastOffset(i, gMulAcc_u8_u16.meta1);
        #endif
        uint a = gMulAcc_u8_u16.input0[a_idx];
        uint b = gMulAcc_u8_u16.input1[b_idx];
        gMulAcc_u8_u16.output0[i] = a * b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_u8_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMulAcc_u8_u32.meta0);
        b_idx = broadcastOffset(i, gMulAcc_u8_u32.meta1);
        #endif
        uint a = gMulAcc_u8_u32.input0[a_idx];
        uint b = gMulAcc_u8_u32.input1[b_idx];
        gMulAcc_u8_u32.output0[i] = a * b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_u8_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMulAcc_u8_u64.meta0);
        b_idx = broadcastOffset(i, gMulAcc_u8_u64.meta1);
        #endif
        uint64_t a = (uint64_t)gMulAcc_u8_u64.input0[a_idx];
        uint64_t b = (uint64_t)gMulAcc_u8_u64.input1[b_idx];
        gMulAcc_u8_u64.output0[i] = a * b;
    }
}
