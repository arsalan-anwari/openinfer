#include "../../../packed_utils.slang"

struct PushConsts {
    uint len;
    int value_i32;
    int64_t value_i64;
};

[[vk::push_constant]] PushConsts pc;

struct FillPackedBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<FillPackedBuffers> gFillPacked_i1;
ParameterBlock<FillPackedBuffers> gFillPacked_i2;
ParameterBlock<FillPackedBuffers> gFillPacked_i4;

__generic<let BITS : int>
void fillPackedSignedImpl(uint3 tid, FillPackedBuffers bufs, PushConsts pc)
{
    uint i = tid.x;
    if (i < pc.len) {
        int value = pc.value_i32;
        uint raw = (uint)value;
        packed_write(bufs.output0, i, BITS, raw);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_i1(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    fillPackedSignedImpl<1>(tid, gFillPacked_i1, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_i2(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    fillPackedSignedImpl<2>(tid, gFillPacked_i2, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_i4(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    fillPackedSignedImpl<4>(tid, gFillPacked_i4, pc);
}
