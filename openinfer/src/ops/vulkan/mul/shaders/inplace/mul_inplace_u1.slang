#include "../../../packed_utils.slang"
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct MulPackedBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<MulPackedBuffers> gMulPacked;

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_u1(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    uint a = packed_read_unsigned(gMulPacked.input0, i, 1);
    uint b = packed_read_unsigned(gMulPacked.input1, i, 1);
    uint y = a * b;
    packed_write(gMulPacked.output0, i, 1, (uint)y);
}
}
