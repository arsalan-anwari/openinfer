#include "../../../packed/packed_utils.slang"
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AbsPackedBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<AbsPackedBuffers> gAbsPacked;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_i2(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    int x = packed_read_signed(gAbsPacked.input0, i, 2);
    int y = x < 0 ? -x : x;
    packed_write(gAbsPacked.output0, i, 2, (uint)y);
}
}
