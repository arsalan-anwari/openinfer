struct PushConsts {
    uint len;
    uint neg_bits;
    uint clamp_bits;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

half relu_half(half x, half negative_slope, half clamp_max) {
    half y = x >= (half)0.0 ? x : x * negative_slope;
    if (y > clamp_max) {
        y = clamp_max;
    }
    return y;
}

struct ReluBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<half, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<half, Std430DataLayout> output0;
};

ParameterBlock<ReluBuffers> gRelu;

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    half x = gRelu.input0[i];
    half negative_slope = (half)asfloat(pc.neg_bits);
    half clamp_max = (half)asfloat(pc.clamp_bits);
    gRelu.output0[i] = relu_half(x, negative_slope, clamp_max);
}
}
