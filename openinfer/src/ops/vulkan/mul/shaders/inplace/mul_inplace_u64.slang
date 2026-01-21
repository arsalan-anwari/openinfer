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

ParameterBlock<MulBuffers<uint64_t>> gMul_uint64_t;

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    gMul_uint64_t.output0[i] =
        gMul_uint64_t.input0[i] * gMul_uint64_t.input1[i];
}
}
