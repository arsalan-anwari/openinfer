struct PushConsts {
    uint len;
    uint value_bits;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct FillBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<half, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<half, Std430DataLayout> output0;
};

ParameterBlock<FillBuffers> gFill;

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    half value = (half)asfloat(pc.value_bits);
    gFill.output0[i] = value;
}
}
