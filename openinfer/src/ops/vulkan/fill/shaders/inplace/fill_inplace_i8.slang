struct PushConsts {
    uint len;
    uint value_bits;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct FillBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<FillBuffers<int>> gFill_int;

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    float value = asfloat(pc.value_bits);
    gFill_int.output0[i] = (int)value;
}
}
