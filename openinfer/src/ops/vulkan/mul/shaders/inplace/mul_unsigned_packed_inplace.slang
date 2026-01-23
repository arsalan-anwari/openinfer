#include "../../../packed_utils.slang"
#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct MulPackedBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
    [[vk::binding(3, 0)]] StructuredBuffer<uint, Std430DataLayout> meta0;
    [[vk::binding(4, 0)]] StructuredBuffer<uint, Std430DataLayout> meta1;
};

ParameterBlock<MulPackedBuffers> gMulPacked_u1;
ParameterBlock<MulPackedBuffers> gMulPacked_u2;
ParameterBlock<MulPackedBuffers> gMulPacked_u4;

__generic<let BITS : int>
void mulPackedUnsignedImpl(uint3 tid, MulPackedBuffers bufs, PushConsts pc)
{
    uint i = tid.x;
    if (i < pc.len) {
        uint a_idx = broadcastOffset(i, bufs.meta0);
        uint b_idx = broadcastOffset(i, bufs.meta1);
        uint a = packed_read_unsigned(bufs.input0, a_idx, BITS);
        uint b = packed_read_unsigned(bufs.input1, b_idx, BITS);
        uint y = a * b;
        packed_write(bufs.output0, i, BITS, y);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_u1(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulPackedUnsignedImpl<1>(tid, gMulPacked_u1, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_u2(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulPackedUnsignedImpl<2>(tid, gMulPacked_u2, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mul_inplace_u4(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    mulPackedUnsignedImpl<4>(tid, gMulPacked_u4, pc);
}
