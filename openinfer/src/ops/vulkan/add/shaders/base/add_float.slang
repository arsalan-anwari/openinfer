#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AddBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
#ifdef OI_BROADCAST
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

ParameterBlock<AddBuffers<half>> gAdd_f16_native;
ParameterBlock<AddBuffers<float>> gAdd_f32;
ParameterBlock<AddBuffers<double>> gAdd_f64;

__generic<typename T, let BLOCK_SIZE : int>
void addImpl(uint3 tid, AddBuffers<T> bufs, PushConsts pc)
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
        bufs.output0[i] = bufs.input0[a_idx] + bufs.input1[b_idx];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<half, 256>(tid, gAdd_f16_native, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<float, 256>(tid, gAdd_f32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_f64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addImpl<double, 256>(tid, gAdd_f64, pc);
}
