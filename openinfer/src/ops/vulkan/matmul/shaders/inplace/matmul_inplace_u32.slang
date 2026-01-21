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

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_inplace_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    uint acc = (uint)0;
    for (uint kk = 0; kk < pc.k; ++kk) {
        acc = acc + (gMatmul_uint.input0[a_row + kk]
            * gMatmul_uint.input1[kk * pc.n + col]);
    }
    gMatmul_uint.output0[idx] = acc;
}
}
