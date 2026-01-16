// Matmul kernel (single source, multiple entry points).
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

ParameterBlock<MatmulBuffers<float>> gMatmul_f32;
ParameterBlock<MatmulBuffers<int>> gMatmul_i8;
ParameterBlock<MatmulBuffers<int>> gMatmul_i16;
ParameterBlock<MatmulBuffers<int>> gMatmul_i32;
ParameterBlock<MatmulBuffers<uint>> gMatmul_u8;
ParameterBlock<MatmulBuffers<uint>> gMatmul_u16;
ParameterBlock<MatmulBuffers<uint>> gMatmul_u32;
ParameterBlock<MatmulBuffers<uint>> gMatmul_bool;
ParameterBlock<MatmulBuffers<int64_t>> gMatmul_i64;
ParameterBlock<MatmulBuffers<uint64_t>> gMatmul_u64;

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

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<float, 256>(tid, gMatmul_f32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<int, 256>(tid, gMatmul_i8, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<int, 256>(tid, gMatmul_i16, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<int, 256>(tid, gMatmul_i32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_u8(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<uint, 256>(tid, gMatmul_u8, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_u16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<uint, 256>(tid, gMatmul_u16, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<uint, 256>(tid, gMatmul_u32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_bool(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulBoolImpl(tid, gMatmul_bool, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<int64_t, 256>(tid, gMatmul_i64, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_u64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<uint64_t, 256>(tid, gMatmul_u64, pc);
}
