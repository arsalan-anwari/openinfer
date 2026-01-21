#include "../../../packed/packed_utils.slang"
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

struct ReluPackedBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<ReluPackedBuffers> gReluPacked;

[shader("compute")]
[numthreads(256, 1, 1)]
void relu_i4(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    int x = packed_read_signed(gReluPacked.input0, i, 4);
    float y = relu_float((float)x, asfloat(pc.neg_bits), asfloat(pc.clamp_bits));
    if (y < -128.0f) {
        y = -128.0f;
    }
    if (y > 127.0f) {
        y = 127.0f;
    }
    int yi = (int)y;
    packed_write(gReluPacked.output0, i, 4, (uint)yi);
}
}
