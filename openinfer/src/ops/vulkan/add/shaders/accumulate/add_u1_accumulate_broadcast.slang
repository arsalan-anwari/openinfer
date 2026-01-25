#define OI_BROADCAST

#include "../../../packed_utils.slang"
#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AddAccPackedBuffersUint {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
#ifdef OI_BROADCAST

    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

struct AddAccPackedBuffersUint64 {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint64_t, Std430DataLayout> output0;
#ifdef OI_BROADCAST

    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

ParameterBlock<AddAccPackedBuffersUint> gAddAcc_u1_u8;
ParameterBlock<AddAccPackedBuffersUint> gAddAcc_u1_u16;
ParameterBlock<AddAccPackedBuffersUint> gAddAcc_u1_u32;
ParameterBlock<AddAccPackedBuffersUint64> gAddAcc_u1_u64;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u1_u8_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gAddAcc_u1_u8.meta0);
        b_idx = broadcastOffset(i, gAddAcc_u1_u8.meta1);
        #endif
        uint a = packed_read_unsigned(gAddAcc_u1_u8.input0, a_idx, 1);
        uint b = packed_read_unsigned(gAddAcc_u1_u8.input1, b_idx, 1);
        gAddAcc_u1_u8.output0[i] = a + b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u1_u16_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gAddAcc_u1_u16.meta0);
        b_idx = broadcastOffset(i, gAddAcc_u1_u16.meta1);
        #endif
        uint a = packed_read_unsigned(gAddAcc_u1_u16.input0, a_idx, 1);
        uint b = packed_read_unsigned(gAddAcc_u1_u16.input1, b_idx, 1);
        gAddAcc_u1_u16.output0[i] = a + b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u1_u32_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gAddAcc_u1_u32.meta0);
        b_idx = broadcastOffset(i, gAddAcc_u1_u32.meta1);
        #endif
        uint a = packed_read_unsigned(gAddAcc_u1_u32.input0, a_idx, 1);
        uint b = packed_read_unsigned(gAddAcc_u1_u32.input1, b_idx, 1);
        gAddAcc_u1_u32.output0[i] = a + b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u1_u64_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
        #ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, gAddAcc_u1_u64.meta0);
        b_idx = broadcastOffset(i, gAddAcc_u1_u64.meta1);
        #endif
        uint64_t a = (uint64_t)packed_read_unsigned(gAddAcc_u1_u64.input0, a_idx, 1);
        uint64_t b = (uint64_t)packed_read_unsigned(gAddAcc_u1_u64.input1, b_idx, 1);
        gAddAcc_u1_u64.output0[i] = a + b;
    }
}
