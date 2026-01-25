#define OI_BROADCAST

#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct MulAccBuffersInt {
    [[vk::binding(0, 0)]] StructuredBuffer<int, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<int, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int, Std430DataLayout> output0;
#ifdef OI_BROADCAST

    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

struct MulAccBuffersInt64 {
    [[vk::binding(0, 0)]] StructuredBuffer<int, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<int, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int64_t, Std430DataLayout> output0;
#ifdef OI_BROADCAST

    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

ParameterBlock<MulAccBuffersInt> gMulAcc_i8_i16;
ParameterBlock<MulAccBuffersInt> gMulAcc_i8_i32;
ParameterBlock<MulAccBuffersInt64> gMulAcc_i8_i64;

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_i8_i16_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMulAcc_i8_i16.meta0);
        b_idx = broadcastOffset(i, gMulAcc_i8_i16.meta1);
        #endif
        int a = gMulAcc_i8_i16.input0[a_idx];
        int b = gMulAcc_i8_i16.input1[b_idx];
        gMulAcc_i8_i16.output0[i] = a * b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_i8_i32_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMulAcc_i8_i32.meta0);
        b_idx = broadcastOffset(i, gMulAcc_i8_i32.meta1);
        #endif
        int a = gMulAcc_i8_i32.input0[a_idx];
        int b = gMulAcc_i8_i32.input1[b_idx];
        gMulAcc_i8_i32.output0[i] = a * b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_i8_i64_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMulAcc_i8_i64.meta0);
        b_idx = broadcastOffset(i, gMulAcc_i8_i64.meta1);
        #endif
        int64_t a = (int64_t)gMulAcc_i8_i64.input0[a_idx];
        int64_t b = (int64_t)gMulAcc_i8_i64.input1[b_idx];
        gMulAcc_i8_i64.output0[i] = a * b;
    }
}
