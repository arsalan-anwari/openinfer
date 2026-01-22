#include "../../../float_utils.slang"
struct PushConsts {
    uint len;
    uint m;
    uint n;
    uint k;
};

[[vk::push_constant]] PushConsts pc;

struct MatmulBuffers {
    [[vk::binding(0, 0)]] ByteAddressBuffer input0;
    [[vk::binding(1, 0)]] ByteAddressBuffer input1;
    [[vk::binding(2, 0)]] RWByteAddressBuffer output0;
};

ParameterBlock<MatmulBuffers> gMatmul;

[shader("compute")]
[numthreads(256, 1, 1)]
void matmul_inplace_f16(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
uint idx = tid.x;
if (idx < pc.len) {
    uint row = idx / pc.n;
    uint col = idx - row * pc.n;
    uint a_row = row * pc.k;
    precise float acc = 0.0f;
    for (uint kk = 0; kk < pc.k; ++kk) {
        float a = read_f16(gMatmul.input0, a_row + kk);
        float b = read_f16(gMatmul.input1, kk * pc.n + col);
        precise float prod = a * b;
        acc = acc + prod;
    }
    write_f16(gMatmul.output0, idx, acc);
}
}
