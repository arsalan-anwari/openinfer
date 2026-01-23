struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct IsFiniteScalarBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<half, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
};

ParameterBlock<IsFiniteScalarBuffers> gIsFinite;

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint ok = 1u;
half max_val = (half)65504.0;
for (uint i = 0; i < pc.len; ++i) {
    half x = gIsFinite.input0[i];
    half abs_x = abs(x);
    if (x != x || abs_x > max_val) {
        ok = 0u;
        break;
    }
}
gIsFinite.output0[0] = ok;
}
