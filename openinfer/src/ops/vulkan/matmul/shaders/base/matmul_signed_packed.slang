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

ParameterBlock<MatmulPackedBuffers> gMatmulPacked_i1;
ParameterBlock<MatmulPackedBuffers> gMatmulPacked_i2;
ParameterBlock<MatmulPackedBuffers> gMatmulPacked_i4;

__generic<let BITS : int>
void matmulPackedSignedImpl(uint3 tid, MatmulPackedBuffers bufs, PushConsts pc) {
    uint idx = tid.x;
    if (idx < pc.len) {
        uint row = idx / pc.n;
        uint col = idx - row * pc.n;
        uint a_row = row * pc.k;
        int acc = 0;
        for (uint kk = 0; kk < pc.k; ++kk) {
            int a = packed_read_signed(bufs.input0, a_row + kk, BITS);
            int b = packed_read_signed(bufs.input1, kk * pc.n + col, BITS);
            acc = acc + (a * b);
        }
        packed_write(bufs.output0, idx, BITS, (uint)acc);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i1(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulPackedSignedImpl<1>(tid, gMatmulPacked_i1, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i2(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulPackedSignedImpl<2>(tid, gMatmulPacked_i2, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i4(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulPackedSignedImpl<4>(tid, gMatmulPacked_i4, pc);
}
