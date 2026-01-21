struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct MulAccBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<uint, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
};

ParameterBlock<MulAccBuffers> gMulAcc;

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_u8_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    uint a = (uint)gMulAcc.input0[i];
    uint b = (uint)gMulAcc.input1[i];
    gMulAcc.output0[i] = a * b;
}
}
