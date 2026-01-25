#define OI_BROADCAST

// Matmul kernel (native float types).
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

ParameterBlock<MatmulBuffers<half>> gMatmul_f16_native;
ParameterBlock<MatmulBuffers<float>> gMatmul_f32;
ParameterBlock<MatmulBuffers<double>> gMatmul_f64;

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
void matmul_f16_native_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<half, 256>(tid, gMatmul_f16_native, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_f32_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<float, 256>(tid, gMatmul_f32, pc);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_f64_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    matmulImpl<double, 256>(tid, gMatmul_f64, pc);
}
