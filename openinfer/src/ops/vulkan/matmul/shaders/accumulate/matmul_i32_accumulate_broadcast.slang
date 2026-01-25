#define OI_BROADCAST

struct PushConsts {
    uint len;
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]] PushConsts pc;

struct MatmulAccBuffersInt64 {
    [[vk::binding(0, 0)]] StructuredBuffer<int, Std430DataLayout> input0;
    [[vk::binding(1, 0)]] StructuredBuffer<int, Std430DataLayout> input1;
    [[vk::binding(2, 0)]] RWStructuredBuffer<int64_t, Std430DataLayout> output0;
};

ParameterBlock<MatmulAccBuffersInt64> gMatmulAcc_i32_i64;

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_i32_i64_broadcast(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    int64_t acc = (int64_t)0;
    for (uint kk = 0; kk < pc.k; ++kk) {
        int64_t a = (int64_t)gMatmulAcc_i32_i64.input0[a_row + kk];
        int64_t b = (int64_t)gMatmulAcc_i32_i64.input1[kk * pc.n + col];
        acc = acc + (a * b);
    }
    gMatmulAcc_i32_i64.output0[idx] = acc;
}
}
