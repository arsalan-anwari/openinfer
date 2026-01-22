#include "../../../float_utils.slang"
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AbsBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<AbsBuffers> gAbs;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_f8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    float x = read_f8(gAbs.input0, i);
    write_f8(gAbs.output0, i, abs(x));
}
}
