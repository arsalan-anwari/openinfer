#include "../../../packed_utils.slang"

struct PushConsts {
    uint len;
    uint alpha_lo;
    uint alpha_hi;
    uint clamp_lo;
    uint clamp_hi;
    uint flags;
};

[[vk::push_constant]] PushConsts pc;

float relu_float(float x, float alpha, float clamp_max) {
    float y = x >= 0.0f ? x : x * alpha;
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
        float y = relu_float((float)x, asfloat(pc.alpha_lo), asfloat(pc.clamp_lo));
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
