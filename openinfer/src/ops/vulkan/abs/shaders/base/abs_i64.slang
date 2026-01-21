struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AbsBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<AbsBuffers<int64_t>> gAbs_int64_t;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    int x = gAbs_int64_t.input0[i];
    gAbs_int64_t.output0[i] = x < 0 ? -x : x;
}
}
