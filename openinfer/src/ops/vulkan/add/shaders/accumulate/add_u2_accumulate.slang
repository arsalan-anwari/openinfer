#include "../../../packed_utils.slang"
#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AddAccPackedBuffersUint {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
};

struct AddAccPackedBuffersUint64 {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint64_t, Std430DataLayout> output0;
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
};

ParameterBlock<AddAccPackedBuffersUint> gAddAcc_u2_u8;
ParameterBlock<AddAccPackedBuffersUint> gAddAcc_u2_u16;
ParameterBlock<AddAccPackedBuffersUint> gAddAcc_u2_u32;
ParameterBlock<AddAccPackedBuffersUint64> gAddAcc_u2_u64;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u2_u8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, gAddAcc_u2_u8.meta0);
        uint b_idx = broadcastOffset(i, gAddAcc_u2_u8.meta1);
        uint a = packed_read_unsigned(gAddAcc_u2_u8.input0, a_idx, 2);
        uint b = packed_read_unsigned(gAddAcc_u2_u8.input1, b_idx, 2);
        gAddAcc_u2_u8.output0[i] = a + b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u2_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, gAddAcc_u2_u16.meta0);
        uint b_idx = broadcastOffset(i, gAddAcc_u2_u16.meta1);
        uint a = packed_read_unsigned(gAddAcc_u2_u16.input0, a_idx, 2);
        uint b = packed_read_unsigned(gAddAcc_u2_u16.input1, b_idx, 2);
        gAddAcc_u2_u16.output0[i] = a + b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u2_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, gAddAcc_u2_u32.meta0);
        uint b_idx = broadcastOffset(i, gAddAcc_u2_u32.meta1);
        uint a = packed_read_unsigned(gAddAcc_u2_u32.input0, a_idx, 2);
        uint b = packed_read_unsigned(gAddAcc_u2_u32.input1, b_idx, 2);
        gAddAcc_u2_u32.output0[i] = a + b;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u2_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, gAddAcc_u2_u64.meta0);
        uint b_idx = broadcastOffset(i, gAddAcc_u2_u64.meta1);
        uint64_t a = (uint64_t)packed_read_unsigned(gAddAcc_u2_u64.input0, a_idx, 2);
        uint64_t b = (uint64_t)packed_read_unsigned(gAddAcc_u2_u64.input1, b_idx, 2);
        gAddAcc_u2_u64.output0[i] = a + b;
    }
}
