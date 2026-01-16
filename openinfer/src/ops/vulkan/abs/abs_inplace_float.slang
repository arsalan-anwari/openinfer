// Abs inplace kernel (float32 only).
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

ParameterBlock<AbsBuffers<float>> gAbs_f32;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        gAbs_f32.output0[i] = abs(gAbs_f32.input0[i]);
    }
}
