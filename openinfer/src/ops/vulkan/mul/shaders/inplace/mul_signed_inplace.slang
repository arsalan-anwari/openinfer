#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct MulBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
#ifdef OI_BROADCAST
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

ParameterBlock<MulBuffers<int>> gMul_i8;
ParameterBlock<MulBuffers<int>> gMul_i16;
ParameterBlock<MulBuffers<int>> gMul_i32;
ParameterBlock<MulBuffers<int64_t>> gMul_i64;

__generic<typename T, let BLOCK_SIZE : int>
void mulImpl(uint3 tid, MulBuffers<T> bufs, PushConsts pc)
    where T : IArithmetic
{
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
#ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, bufs.meta0);
        b_idx = broadcastOffset(i, bufs.meta1);
#endif
        bufs.output0[i] = bufs.input0[a_idx] * bufs.input1[b_idx];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<int, 256>(tid, gMul_i8, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_i16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<int, 256>(tid, gMul_i16, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<int, 256>(tid, gMul_i32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<int64_t, 256>(tid, gMul_i64, pc);
}
