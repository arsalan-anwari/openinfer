struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AddBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<AddBuffers<float>> gAdd_float;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    gAdd_float.output0[i] =
        gAdd_float.input0[i] + gAdd_float.input1[i];
}
}
