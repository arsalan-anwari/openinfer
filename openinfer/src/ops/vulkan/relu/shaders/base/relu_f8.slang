#include "../../../float_utils.slang"
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

struct ReluBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<ReluBuffers> gRelu;

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_f8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    float x = read_f8(gRelu.input0, i);
    float y = relu_float(x, asfloat(pc.neg_bits), asfloat(pc.clamp_bits));
    write_f8(gRelu.output0, i, y);
}
}
