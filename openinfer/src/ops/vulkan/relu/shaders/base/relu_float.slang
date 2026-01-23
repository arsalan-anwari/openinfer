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

ParameterBlock<ReluBuffers<half>> gRelu_f16_native;
ParameterBlock<ReluBuffers<float>> gRelu_f32;
ParameterBlock<ReluBuffers<double>> gRelu_f64;

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        half x = gRelu_f16_native.input0[i];
        float y = relu_float((float)x, asfloat(pc.neg_bits), asfloat(pc.clamp_bits));
        gRelu_f16_native.output0[i] = (half)y;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float x = gRelu_f32.input0[i];
        float y = relu_float(x, asfloat(pc.neg_bits), asfloat(pc.clamp_bits));
        gRelu_f32.output0[i] = y;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_f64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        double x = gRelu_f64.input0[i];
        float y = relu_float((float)x, asfloat(pc.neg_bits), asfloat(pc.clamp_bits));
        gRelu_f64.output0[i] = (double)y;
    }
}
