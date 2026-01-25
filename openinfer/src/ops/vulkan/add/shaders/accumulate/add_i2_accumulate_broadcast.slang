#define OI_BROADCAST

#include "../../../packed_utils.slang"
#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AddAccPackedBuffersInt {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int, Std430DataLayout> output0;
#ifdef OI_BROADCAST
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

struct AddAccPackedBuffersInt64 {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int64_t, Std430DataLayout> output0;
#ifdef OI_BROADCAST
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

ParameterBlock<AddAccPackedBuffersInt> gAddAcc_i2_i8;
ParameterBlock<AddAccPackedBuffersInt> gAddAcc_i2_i16;
ParameterBlock<AddAccPackedBuffersInt> gAddAcc_i2_i32;
ParameterBlock<AddAccPackedBuffersInt64> gAddAcc_i2_i64;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i2_i8_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
#ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gAddAcc_i2_i8.meta0);
        b_idx = broadcastOffset(i, gAddAcc_i2_i8.meta1);
#endif
        int a = packed_read_signed(gAddAcc_i2_i8.input0, a_idx, 2);
        int b = packed_read_signed(gAddAcc_i2_i8.input1, b_idx, 2);
        gAddAcc_i2_i8.output0[i] = a + b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i2_i16_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
#ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gAddAcc_i2_i16.meta0);
        b_idx = broadcastOffset(i, gAddAcc_i2_i16.meta1);
#endif
        int a = packed_read_signed(gAddAcc_i2_i16.input0, a_idx, 2);
        int b = packed_read_signed(gAddAcc_i2_i16.input1, b_idx, 2);
        gAddAcc_i2_i16.output0[i] = a + b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i2_i32_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
#ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gAddAcc_i2_i32.meta0);
        b_idx = broadcastOffset(i, gAddAcc_i2_i32.meta1);
#endif
        int a = packed_read_signed(gAddAcc_i2_i32.input0, a_idx, 2);
        int b = packed_read_signed(gAddAcc_i2_i32.input1, b_idx, 2);
        gAddAcc_i2_i32.output0[i] = a + b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i2_i64_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
#ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gAddAcc_i2_i64.meta0);
        b_idx = broadcastOffset(i, gAddAcc_i2_i64.meta1);
#endif
        int64_t a = (int64_t)packed_read_signed(gAddAcc_i2_i64.input0, a_idx, 2);
        int64_t b = (int64_t)packed_read_signed(gAddAcc_i2_i64.input1, b_idx, 2);
        gAddAcc_i2_i64.output0[i] = a + b;
    }
}
