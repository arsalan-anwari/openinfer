// Relu kernel (float32 only).
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct ReluBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<ReluBuffers<float>> gRelu_f32;

float reluFloat(float x, float negative_slope, float clamp_max) {
    float y = (x >= 0.0f) ? x : (x * negative_slope);
    if (y > clamp_max) {
        y = clamp_max;
    }
    return y;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float x = gRelu_f32.input0[i];
        float negative_slope = asfloat(pc.flags);
        float clamp_max = asfloat(pc.pad0);
        float y = reluFloat(x, negative_slope, clamp_max);
        gRelu_f32.output0[i] = y;
    }
}
