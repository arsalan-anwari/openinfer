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

ParameterBlock<MatmulBuffers<double>> gMatmul_double;

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_f64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    double acc = (double)0;
    for (uint kk = 0; kk < pc.k; ++kk) {
        acc = acc + (gMatmul_double.input0[a_row + kk]
            * gMatmul_double.input1[kk * pc.n + col]);
    }
    gMatmul_double.output0[idx] = acc;
}
}
