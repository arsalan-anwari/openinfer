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

ParameterBlock<MulBuffers<uint>> gMul_uint;

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_bitset(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    uint a = gMul_uint.input0[i] & 0xFFu;
    uint b = gMul_uint.input1[i] & 0xFFu;
    gMul_uint.output0[i] = (a * b) & 0xFFu;
}
}
