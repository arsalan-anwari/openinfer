#include "../../../packed_utils.slang"
struct PushConsts {
    uint len;
    uint flags;
    uint pad0;
    uint pad1;
};

[[vk::push_constant]] PushConsts pc;

struct AddPackedBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<AddPackedBuffers> gAddPacked;

[shader("compute")]
[numthreads(256, 1, 1)]
void add_inplace_i1(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint i = tid.x;
if (i < pc.len) {
    int a = packed_read_signed(gAddPacked.input0, i, 1);
    int b = packed_read_signed(gAddPacked.input1, i, 1);
    int y = a + b;
    packed_write(gAddPacked.output0, i, 1, (uint)y);
}
}
