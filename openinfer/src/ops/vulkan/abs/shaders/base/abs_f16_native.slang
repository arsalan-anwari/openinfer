struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AbsBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<half, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<half, Std430DataLayout> output0;
};

ParameterBlock<AbsBuffers> gAbs;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    half x = gAbs.input0[i];
    gAbs.output0[i] = abs(x);
}
}
