struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct MulAccBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<int, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<int, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int, Std430DataLayout> output0;
};

ParameterBlock<MulAccBuffers> gMulAcc;

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_i8_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    int a = (int)gMulAcc.input0[i];
    int b = (int)gMulAcc.input1[i];
    gMulAcc.output0[i] = a * b;
}
}
