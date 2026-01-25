#define OI_BROADCAST

#include "../../../packed_utils.slang"
struct PushConsts {
    uint len;
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]] PushConsts pc;

struct MatmulAccBuffersInt {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int, Std430DataLayout> output0;
};

struct MatmulAccBuffersInt64 {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int64_t, Std430DataLayout> output0;
};

ParameterBlock<MatmulAccBuffersInt> gMatmulAcc_i2_i8;
ParameterBlock<MatmulAccBuffersInt> gMatmulAcc_i2_i16;
ParameterBlock<MatmulAccBuffersInt> gMatmulAcc_i2_i32;
ParameterBlock<MatmulAccBuffersInt64> gMatmulAcc_i2_i64;

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i2_i8_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    int acc = (int)0;
    for (uint kk = 0; kk < pc.k; ++kk) {
        int a = packed_read_signed(gMatmulAcc_i2_i8.input0, a_row + kk, 2);
        int b = packed_read_signed(gMatmulAcc_i2_i8.input1, kk * pc.n + col, 2);
        acc = acc + (a * b);
    }
    gMatmulAcc_i2_i8.output0[idx] = acc;
}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i2_i16_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    int acc = (int)0;
    for (uint kk = 0; kk < pc.k; ++kk) {
        int a = packed_read_signed(gMatmulAcc_i2_i16.input0, a_row + kk, 2);
        int b = packed_read_signed(gMatmulAcc_i2_i16.input1, kk * pc.n + col, 2);
        acc = acc + (a * b);
    }
    gMatmulAcc_i2_i16.output0[idx] = acc;
}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i2_i32_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    int acc = (int)0;
    for (uint kk = 0; kk < pc.k; ++kk) {
        int a = packed_read_signed(gMatmulAcc_i2_i32.input0, a_row + kk, 2);
        int b = packed_read_signed(gMatmulAcc_i2_i32.input1, kk * pc.n + col, 2);
        acc = acc + (a * b);
    }
    gMatmulAcc_i2_i32.output0[idx] = acc;
}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i2_i64_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    int64_t acc = (int64_t)0;
    for (uint kk = 0; kk < pc.k; ++kk) {
        int64_t a = (int64_t)packed_read_signed(gMatmulAcc_i2_i64.input0, a_row + kk, 2);
        int64_t b = (int64_t)packed_read_signed(gMatmulAcc_i2_i64.input1, kk * pc.n + col, 2);
        acc = acc + (a * b);
    }
    gMatmulAcc_i2_i64.output0[idx] = acc;
}
}
