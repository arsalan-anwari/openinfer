#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct MulBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
};

ParameterBlock<MulBuffers<uint>> gMul_u8;
ParameterBlock<MulBuffers<uint>> gMul_u16;
ParameterBlock<MulBuffers<uint>> gMul_u32;
ParameterBlock<MulBuffers<uint64_t>> gMul_u64;
ParameterBlock<MulBuffers<uint>> gMul_bool;
ParameterBlock<MulBuffers<uint>> gMul_bitset;

__generic<typename T, let BLOCK_SIZE : int>
void mulImpl(uint3 tid, MulBuffers<T> bufs, PushConsts pc)
    where T : IArithmetic
{
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, bufs.meta0);
        uint b_idx = broadcastOffset(i, bufs.meta1);
        bufs.output0[i] = bufs.input0[a_idx] * bufs.input1[b_idx];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_u8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<uint, 256>(tid, gMul_u8, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<uint, 256>(tid, gMul_u16, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<uint, 256>(tid, gMul_u32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<uint64_t, 256>(tid, gMul_u64, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_bool(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, gMul_bool.meta0);
        uint b_idx = broadcastOffset(i, gMul_bool.meta1);
        uint a = gMul_bool.input0[a_idx];
        uint b = gMul_bool.input1[b_idx];
        gMul_bool.output0[i] = (a != 0u && b != 0u) ? 1u : 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_bitset(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, gMul_bitset.meta0);
        uint b_idx = broadcastOffset(i, gMul_bitset.meta1);
        uint a = gMul_bitset.input0[a_idx] & 0xFFu;
        uint b = gMul_bitset.input1[b_idx] & 0xFFu;
        gMul_bitset.output0[i] = (a * b) & 0xFFu;
    }
}
