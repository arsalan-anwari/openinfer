#include "../../../packed_utils.slang"
#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AbsPackedBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<AbsPackedBuffers> gAbsPacked_i1;
ParameterBlock<AbsPackedBuffers> gAbsPacked_i2;
ParameterBlock<AbsPackedBuffers> gAbsPacked_i4;

__generic<let BITS : int>
void absPackedImpl(uint3 tid, AbsPackedBuffers bufs, PushConsts pc)
{
    uint i = tid.x;
    if (i < pc.len) {
        int x = packed_read_signed(bufs.input0, i, BITS);
        int y = x < 0 ? -x : x;
        packed_write(bufs.output0, i, BITS, (uint)y);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_i1(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    absPackedImpl<1>(tid, gAbsPacked_i1, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_i2(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    absPackedImpl<2>(tid, gAbsPacked_i2, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_inplace_i4(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    absPackedImpl<4>(tid, gAbsPacked_i4, pc);
}
