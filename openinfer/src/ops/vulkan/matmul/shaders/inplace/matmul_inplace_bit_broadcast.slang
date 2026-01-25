#define OI_BROADCAST

struct PushConsts {
    uint len;
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]] PushConsts pc;

struct MatmulBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<T, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<MatmulBuffers<uint>> gMatmul_uint;

void matmulBoolImpl(uint3 tid, MatmulBuffers<uint> bufs, PushConsts pc) {
    uint idx = tid.x;
    if (idx < pc.len) {
        uint row = idx / pc.n;
        uint col = idx - row * pc.n;
        uint a_row = row * pc.k;
        uint acc = 0;
        for (uint kk = 0; kk < pc.k; ++kk) {
            uint a = bufs.input0[a_row + kk];
            uint b = bufs.input1[kk * pc.n + col];
            if (a != 0 && b != 0) {
                acc = 1;
                break;
            }
        }
        bufs.output0[idx] = acc;
    }
}

void matmulBitsetImpl(uint3 tid, MatmulBuffers<uint> bufs, PushConsts pc) {
    uint idx = tid.x;
    if (idx < pc.len) {
        uint row = idx / pc.n;
        uint col = idx - row * pc.n;
        uint a_row = row * pc.k;
        uint acc = 0u;
        for (uint kk = 0; kk < pc.k; ++kk) {
            uint a = bufs.input0[a_row + kk] & 0xFFu;
            uint b = bufs.input1[kk * pc.n + col] & 0xFFu;
            acc = (acc + (a * b)) & 0xFFu;
        }
        bufs.output0[idx] = acc;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_inplace_bool_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulBoolImpl(tid, gMatmul_uint, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_inplace_bitset_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulBitsetImpl(tid, gMatmul_uint, pc);
}
