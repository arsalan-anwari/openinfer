#include "../../../packed/packed_utils.slang"
struct PushConsts {
    uint len;
    uint value_bits;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct FillPackedBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<FillPackedBuffers> gFillPacked;

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_u2(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    float value = asfloat(pc.value_bits);
    uint raw = (uint)((int)value);
    packed_write(gFillPacked.output0, i, 2, raw);
}
}
