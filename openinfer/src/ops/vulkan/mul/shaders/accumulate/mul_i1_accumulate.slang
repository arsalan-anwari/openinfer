#include "../../../packed_utils.slang"
#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct MulAccPackedBuffersInt {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int, Std430DataLayout> output0;
#ifdef OI_BROADCAST

    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

struct MulAccPackedBuffersInt64 {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int64_t, Std430DataLayout> output0;
#ifdef OI_BROADCAST

    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

ParameterBlock<MulAccPackedBuffersInt> gMulAcc_i1_i8;
ParameterBlock<MulAccPackedBuffersInt> gMulAcc_i1_i16;
ParameterBlock<MulAccPackedBuffersInt> gMulAcc_i1_i32;
ParameterBlock<MulAccPackedBuffersInt64> gMulAcc_i1_i64;

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_i1_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMulAcc_i1_i8.meta0);
        b_idx = broadcastOffset(i, gMulAcc_i1_i8.meta1);
        #endif
        int a = packed_read_signed(gMulAcc_i1_i8.input0, a_idx, 1);
        int b = packed_read_signed(gMulAcc_i1_i8.input1, b_idx, 1);
        gMulAcc_i1_i8.output0[i] = a * b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_i1_i16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMulAcc_i1_i16.meta0);
        b_idx = broadcastOffset(i, gMulAcc_i1_i16.meta1);
        #endif
        int a = packed_read_signed(gMulAcc_i1_i16.input0, a_idx, 1);
        int b = packed_read_signed(gMulAcc_i1_i16.input1, b_idx, 1);
        gMulAcc_i1_i16.output0[i] = a * b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_i1_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMulAcc_i1_i32.meta0);
        b_idx = broadcastOffset(i, gMulAcc_i1_i32.meta1);
        #endif
        int a = packed_read_signed(gMulAcc_i1_i32.input0, a_idx, 1);
        int b = packed_read_signed(gMulAcc_i1_i32.input1, b_idx, 1);
        gMulAcc_i1_i32.output0[i] = a * b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_i1_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gMulAcc_i1_i64.meta0);
        b_idx = broadcastOffset(i, gMulAcc_i1_i64.meta1);
        #endif
        int64_t a = (int64_t)packed_read_signed(gMulAcc_i1_i64.input0, a_idx, 1);
        int64_t b = (int64_t)packed_read_signed(gMulAcc_i1_i64.input1, b_idx, 1);
        gMulAcc_i1_i64.output0[i] = a * b;
    }
}
