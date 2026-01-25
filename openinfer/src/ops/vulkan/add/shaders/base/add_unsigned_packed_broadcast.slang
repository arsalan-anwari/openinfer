#define OI_BROADCAST

#include "../../../packed_utils.slang"
#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AddPackedBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
#ifdef OI_BROADCAST
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
#endif
};

ParameterBlock<AddPackedBuffers> gAddPacked_u1;
ParameterBlock<AddPackedBuffers> gAddPacked_u2;
ParameterBlock<AddPackedBuffers> gAddPacked_u4;

__generic<let BITS : int>
void addPackedUnsignedImpl(uint3 tid, AddPackedBuffers bufs, PushConsts pc)
{
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = i;
        uint b_idx = i;
#ifdef OI_BROADCAST
        a_idx = broadcastOffset(i, bufs.meta0);
        b_idx = broadcastOffset(i, bufs.meta1);
#endif
        uint a = packed_read_unsigned(bufs.input0, a_idx, BITS);
        uint b = packed_read_unsigned(bufs.input1, b_idx, BITS);
        uint y = a + b;
        packed_write(bufs.output0, i, BITS, y);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u1_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addPackedUnsignedImpl<1>(tid, gAddPacked_u1, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u2_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addPackedUnsignedImpl<2>(tid, gAddPacked_u2, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void add_u4_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    addPackedUnsignedImpl<4>(tid, gAddPacked_u4, pc);
}
