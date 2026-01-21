struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AddAccBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<uint, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
};

ParameterBlock<AddAccBuffers> gAddAcc;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u16_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    uint a = (uint)gAddAcc.input0[i];
    uint b = (uint)gAddAcc.input1[i];
    gAddAcc.output0[i] = a + b;
}
}
