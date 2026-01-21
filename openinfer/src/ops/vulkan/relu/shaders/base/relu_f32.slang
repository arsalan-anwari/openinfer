struct PushConsts {
    uint len;
    uint neg_bits;
    uint clamp_bits;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

float relu_float(float x, float negative_slope, float clamp_max) {
    float y = x >= 0.0f ? x : x * negative_slope;
    if (y > clamp_max) {
        y = clamp_max;
    }
    return y;
}

struct ReluBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<ReluBuffers<float>> gRelu_float;

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    float x = gRelu_float.input0[i];
    float y = relu_float((float)x, asfloat(pc.neg_bits), asfloat(pc.clamp_bits));
    gRelu_float.output0[i] = (float)y;
}
}
