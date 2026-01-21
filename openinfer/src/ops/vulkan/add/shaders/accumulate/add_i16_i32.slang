struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AddAccBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<int, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<int, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int, Std430DataLayout> output0;
};

ParameterBlock<AddAccBuffers> gAddAcc;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i16_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    int a = (int)gAddAcc.input0[i];
    int b = (int)gAddAcc.input1[i];
    gAddAcc.output0[i] = a + b;
}
}
