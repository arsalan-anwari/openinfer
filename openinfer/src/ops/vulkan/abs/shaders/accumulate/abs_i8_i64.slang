struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AbsAccBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<int, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int64_t, Std430DataLayout> output0;
};

ParameterBlock<AbsAccBuffers> gAbsAcc;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_i8_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    int64_t x = (int64_t)gAbsAcc.input0[i];
    gAbsAcc.output0[i] = x < 0 ? -x : x;
}
}
