#define OI_BROADCAST

// Matmul kernel (signed integers).
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

ParameterBlock<MatmulBuffers<int>> gMatmul_i8;
ParameterBlock<MatmulBuffers<int>> gMatmul_i16;
ParameterBlock<MatmulBuffers<int>> gMatmul_i32;
ParameterBlock<MatmulBuffers<int64_t>> gMatmul_i64;

__generic<typename T, let BLOCK_SIZE : int>
void matmulImpl(uint3 tid, MatmulBuffers<T> bufs, PushConsts pc)
    where T : IArithmetic
{
    uint idx = tid.x;
    if (idx < pc.len) {
        uint row = idx / pc.n;
        uint col = idx - row * pc.n;
        uint a_row = row * pc.k;
        T acc = (T)0;
        for (uint kk = 0; kk < pc.k; ++kk) {
            acc = acc + (bufs.input0[a_row + kk] * bufs.input1[kk * pc.n + col]);
        }
        bufs.output0[idx] = acc;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i8_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<int, 256>(tid, gMatmul_i8, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i16_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<int, 256>(tid, gMatmul_i16, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i32_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<int, 256>(tid, gMatmul_i32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i64_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<int64_t, 256>(tid, gMatmul_i64, pc);
}

