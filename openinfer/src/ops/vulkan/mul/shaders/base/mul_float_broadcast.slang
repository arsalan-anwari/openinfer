#define OI_BROADCAST

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

ParameterBlock<MulBuffers<half>> gMul_f16_native;
ParameterBlock<MulBuffers<float>> gMul_f32;
ParameterBlock<MulBuffers<double>> gMul_f64;

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
void mul_f16_native_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<half, 256>(tid, gMul_f16_native, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_f32_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<float, 256>(tid, gMul_f32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_f64_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<double, 256>(tid, gMul_f64, pc);
}
