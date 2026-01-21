struct PushConsts {
    uint len;
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]] PushConsts pc;

struct MatmulAccBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<uint, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<uint, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> output0;
};

ParameterBlock<MatmulAccBuffers> gMatmulAcc;

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_u16_u32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    uint acc = (uint)0;
    for (uint kk = 0; kk < pc.k; ++kk) {
        uint a = (uint)gMatmulAcc.input0[a_row + kk];
        uint b = (uint)gMatmulAcc.input1[kk * pc.n + col];
        acc = acc + (a * b);
    }
    gMatmulAcc.output0[idx] = acc;
}
}
