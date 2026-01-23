#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct MulBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
};

ParameterBlock<MulBuffers<half>> gMul_f16_native;
ParameterBlock<MulBuffers<float>> gMul_f32;
ParameterBlock<MulBuffers<double>> gMul_f64;

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
void mul_inplace_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<half, 256>(tid, gMul_f16_native, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<float, 256>(tid, gMul_f32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_f64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<double, 256>(tid, gMul_f64, pc);
}
