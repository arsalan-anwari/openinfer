#include "../../../packed_utils.slang"
#include "../../../op_common.slang"

[[vk::push_constant]] PushConsts pc;

struct AbsAccPackedBuffersInt {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int, Std430DataLayout> output0;
};

struct AbsAccPackedBuffersInt64 {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int64_t, Std430DataLayout> output0;
};

ParameterBlock<AbsAccPackedBuffersInt> gAbsAcc_i4_i8;
ParameterBlock<AbsAccPackedBuffersInt> gAbsAcc_i4_i16;
ParameterBlock<AbsAccPackedBuffersInt> gAbsAcc_i4_i32;
ParameterBlock<AbsAccPackedBuffersInt64> gAbsAcc_i4_i64;

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_i4_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int x = packed_read_signed(gAbsAcc_i4_i8.input0, i, 4);
        int y = x < 0 ? -x : x;
        gAbsAcc_i4_i8.output0[i] = y;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_i4_i16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int x = packed_read_signed(gAbsAcc_i4_i16.input0, i, 4);
        int y = x < 0 ? -x : x;
        gAbsAcc_i4_i16.output0[i] = y;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_i4_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int x = packed_read_signed(gAbsAcc_i4_i32.input0, i, 4);
        int y = x < 0 ? -x : x;
        gAbsAcc_i4_i32.output0[i] = y;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void abs_i4_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        int64_t x = (int64_t)packed_read_signed(gAbsAcc_i4_i64.input0, i, 4);
        int64_t y = x < 0 ? -x : x;
        gAbsAcc_i4_i64.output0[i] = y;
    }
}
