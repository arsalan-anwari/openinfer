#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AddBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
};

ParameterBlock<AddBuffers<uint>> gAdd_u8;
ParameterBlock<AddBuffers<uint>> gAdd_u16;
ParameterBlock<AddBuffers<uint>> gAdd_u32;
ParameterBlock<AddBuffers<uint64_t>> gAdd_u64;
ParameterBlock<AddBuffers<uint>> gAdd_bool;
ParameterBlock<AddBuffers<uint>> gAdd_bitset;

__generic<typename T, let BLOCK_SIZE : int>
void addImpl(uint3 tid, AddBuffers<T> bufs, PushConsts pc)
    where T : IArithmetic
{
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, bufs.meta0);
        uint b_idx = broadcastOffset(i, bufs.meta1);
        bufs.output0[i] = bufs.input0[a_idx] + bufs.input1[b_idx];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<uint, 256>(tid, gAdd_u8, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<uint, 256>(tid, gAdd_u16, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<uint, 256>(tid, gAdd_u32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<uint64_t, 256>(tid, gAdd_u64, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_bool(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, gAdd_bool.meta0);
        uint b_idx = broadcastOffset(i, gAdd_bool.meta1);
        uint a = gAdd_bool.input0[a_idx];
        uint b = gAdd_bool.input1[b_idx];
        gAdd_bool.output0[i] = (a != 0u || b != 0u) ? 1u : 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_bitset(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, gAdd_bitset.meta0);
        uint b_idx = broadcastOffset(i, gAdd_bitset.meta1);
        uint a = gAdd_bitset.input0[a_idx] & 0xFFu;
        uint b = gAdd_bitset.input1[b_idx] & 0xFFu;
        gAdd_bitset.output0[i] = (a + b) & 0xFFu;
    }
}
