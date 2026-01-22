#include "../../../float_utils.slang"
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct MulBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<MulBuffers> gMul;

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_f8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    float a = read_f8(gMul.input0, i);
    float b = read_f8(gMul.input1, i);
    float y = a * b;
    write_f8(gMul.output0, i, y);
}
}
