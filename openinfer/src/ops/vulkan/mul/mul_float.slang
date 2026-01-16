// Mul kernel (float32 only).
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct MulBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<MulBuffers<float>> gMul_f32;

__generic<typename T, let BLOCK_SIZE : int>
void mulImpl(uint3 tid, MulBuffers<T> bufs, PushConsts pc)
    where T : IArithmetic
{
    uint i = tid.x;
    if (i < pc.len) {
        bufs.output0[i] = bufs.input0[i] * bufs.input1[i];
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulImpl<float, 256>(tid, gMul_f32, pc);
}
