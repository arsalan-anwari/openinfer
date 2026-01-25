#define OI_BROADCAST

#include "../../../packed_utils.slang"
struct PushConsts {
    uint len;
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]] PushConsts pc;

struct MatmulAccBuffersUint {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
};

struct MatmulAccBuffersUint64 {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint64_t, Std430DataLayout> output0;
};

ParameterBlock<MatmulAccBuffersUint> gMatmulAcc_u2_u8;
ParameterBlock<MatmulAccBuffersUint> gMatmulAcc_u2_u16;
ParameterBlock<MatmulAccBuffersUint> gMatmulAcc_u2_u32;
ParameterBlock<MatmulAccBuffersUint64> gMatmulAcc_u2_u64;

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_u2_u8_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    uint acc = 0u;
    for (uint kk = 0; kk < pc.k; ++kk) {
        uint a = packed_read_unsigned(gMatmulAcc_u2_u8.input0, a_row + kk, 2);
        uint b = packed_read_unsigned(gMatmulAcc_u2_u8.input1, kk * pc.n + col, 2);
        acc = acc + (a * b);
    }
    gMatmulAcc_u2_u8.output0[idx] = acc;
}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_u2_u16_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    uint acc = 0u;
    for (uint kk = 0; kk < pc.k; ++kk) {
        uint a = packed_read_unsigned(gMatmulAcc_u2_u16.input0, a_row + kk, 2);
        uint b = packed_read_unsigned(gMatmulAcc_u2_u16.input1, kk * pc.n + col, 2);
        acc = acc + (a * b);
    }
    gMatmulAcc_u2_u16.output0[idx] = acc;
}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_u2_u32_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    uint acc = 0u;
    for (uint kk = 0; kk < pc.k; ++kk) {
        uint a = packed_read_unsigned(gMatmulAcc_u2_u32.input0, a_row + kk, 2);
        uint b = packed_read_unsigned(gMatmulAcc_u2_u32.input1, kk * pc.n + col, 2);
        acc = acc + (a * b);
    }
    gMatmulAcc_u2_u32.output0[idx] = acc;
}
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_u2_u64_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    uint64_t acc = (uint64_t)0;
    for (uint kk = 0; kk < pc.k; ++kk) {
        uint64_t a = (uint64_t)packed_read_unsigned(gMatmulAcc_u2_u64.input0, a_row + kk, 2);
        uint64_t b = (uint64_t)packed_read_unsigned(gMatmulAcc_u2_u64.input1, kk * pc.n + col, 2);
        acc = acc + (a * b);
    }
    gMatmulAcc_u2_u64.output0[idx] = acc;
}
}
