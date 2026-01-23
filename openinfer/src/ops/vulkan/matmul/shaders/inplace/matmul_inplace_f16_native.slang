struct PushConsts {
    uint len;
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]] PushConsts pc;

struct MatmulBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<half, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<half, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<half, Std430DataLayout> output0;
};

ParameterBlock<MatmulBuffers> gMatmul;

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_inplace_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    half acc = (half)0.0;
    for (uint kk = 0; kk < pc.k; ++kk) {
        half a = gMatmul.input0[a_row + kk];
        half b = gMatmul.input1[kk * pc.n + col];
        acc = acc + a * b;
    }
    gMatmul.output0[idx] = acc;
}
}
