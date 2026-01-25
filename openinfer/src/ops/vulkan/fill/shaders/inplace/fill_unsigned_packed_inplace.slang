#include "../../../packed_utils.slang"

struct PushConsts {
    uint len;
    uint value_u32;
    uint64_t value_u64;
};

[[vk::push_constant]] PushConsts pc;

struct FillPackedBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<FillPackedBuffers> gFillPacked_u1;
ParameterBlock<FillPackedBuffers> gFillPacked_u2;
ParameterBlock<FillPackedBuffers> gFillPacked_u4;

__generic<let BITS : int>
void fillPackedUnsignedImpl(uint3 tid, FillPackedBuffers bufs, PushConsts pc)
{
    uint i = tid.x;
    if (i < pc.len) {
        uint raw = pc.value_u32;
        packed_write(bufs.output0, i, BITS, raw);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_u1(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    fillPackedUnsignedImpl<1>(tid, gFillPacked_u1, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_u2(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    fillPackedUnsignedImpl<2>(tid, gFillPacked_u2, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_u4(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    fillPackedUnsignedImpl<4>(tid, gFillPacked_u4, pc);
}
