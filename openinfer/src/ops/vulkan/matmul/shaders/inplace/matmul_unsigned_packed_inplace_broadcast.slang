#define OI_BROADCAST

#include "../../../packed_utils.slang"
struct PushConsts {
    uint len;
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]] PushConsts pc;

struct MatmulPackedBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<MatmulPackedBuffers> gMatmulPacked_u1;
ParameterBlock<MatmulPackedBuffers> gMatmulPacked_u2;
ParameterBlock<MatmulPackedBuffers> gMatmulPacked_u4;

__generic<let BITS : int>
void matmulPackedUnsignedImpl(uint3 tid, MatmulPackedBuffers bufs, PushConsts pc) {
    uint idx = tid.x;
    if (idx < pc.len) {
        uint row = idx / pc.n;
        uint col = idx - row * pc.n;
        uint a_row = row * pc.k;
        uint acc = 0u;
        for (uint kk = 0; kk < pc.k; ++kk) {
            uint a = packed_read_unsigned(bufs.input0, a_row + kk, BITS);
            uint b = packed_read_unsigned(bufs.input1, kk * pc.n + col, BITS);
            acc = acc + (a * b);
        }
        packed_write(bufs.output0, idx, BITS, acc);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_inplace_u1_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulPackedUnsignedImpl<1>(tid, gMatmulPacked_u1, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_inplace_u2_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulPackedUnsignedImpl<2>(tid, gMatmulPacked_u2, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_inplace_u4_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulPackedUnsignedImpl<4>(tid, gMatmulPacked_u4, pc);
}
