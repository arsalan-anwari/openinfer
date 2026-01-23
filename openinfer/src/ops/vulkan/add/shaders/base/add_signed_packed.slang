#include "../../../packed_utils.slang"
#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AddPackedBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
};

ParameterBlock<AddPackedBuffers> gAddPacked_i1;
ParameterBlock<AddPackedBuffers> gAddPacked_i2;
ParameterBlock<AddPackedBuffers> gAddPacked_i4;

__generic<let BITS : int>
void addPackedSignedImpl(uint3 tid, AddPackedBuffers bufs, PushConsts pc)
{
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, bufs.meta0);
        uint b_idx = broadcastOffset(i, bufs.meta1);
        int a = packed_read_signed(bufs.input0, a_idx, BITS);
        int b = packed_read_signed(bufs.input1, b_idx, BITS);
        int y = a + b;
        packed_write(bufs.output0, i, BITS, (uint)y);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i1(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addPackedSignedImpl<1>(tid, gAddPacked_i1, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i2(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addPackedSignedImpl<2>(tid, gAddPacked_i2, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_i4(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addPackedSignedImpl<4>(tid, gAddPacked_i4, pc);
}
