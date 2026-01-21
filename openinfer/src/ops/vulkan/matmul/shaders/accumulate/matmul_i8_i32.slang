struct PushConsts {
    uint len;
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]] PushConsts pc;

struct MatmulAccBuffers {
    [[vk::binding(0, 0)]] StructuredBuffer<int, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<int, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int, Std430DataLayout> output0;
};

ParameterBlock<MatmulAccBuffers> gMatmulAcc;

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i8_i32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    int acc = (int)0;
    for (uint kk = 0; kk < pc.k; ++kk) {
        int a = (int)gMatmulAcc.input0[a_row + kk];
        int b = (int)gMatmulAcc.input1[kk * pc.n + col];
        acc = acc + (a * b);
    }
    gMatmulAcc.output0[idx] = acc;
}
}
