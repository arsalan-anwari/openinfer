// Is-finite kernel (float32 only).
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct IsFiniteScalarBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
};

ParameterBlock<IsFiniteScalarBuffers<float>> gIsFinite_f32;

[shader("compute")]
[numthreads(1, 1, 1)]
void is_finite_scalar_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint ok = 1u;
    for (uint i = 0; i < pc.len; ++i) {
        float x = gIsFinite_f32.input0[i];
        float abs_x = abs(x);
        if (x != x || abs_x > 3.402823466e38f) {
            ok = 0u;
            break;
        }
    }
    gIsFinite_f32.output0[0] = ok;
}
