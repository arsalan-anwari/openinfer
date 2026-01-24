struct PushConsts {
    uint len;
    uint value_bits_lo;
    uint value_bits_hi;
    uint use_f64_bits;
};

[[vk::push_constant]] PushConsts pc;

struct FillBuffers<T> {
    [[vk::binding(0, 0)]] StructuredBuffer<T, Std430DataLayout> input0;
    [[vk::binding(2, 0)]] RWStructuredBuffer<T, Std430DataLayout> output0;
};

ParameterBlock<FillBuffers<half>> gFill_f16_native;
ParameterBlock<FillBuffers<float>> gFill_f32;
ParameterBlock<FillBuffers<double>> gFill_f64;

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_f16_native(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits_lo);
        gFill_f16_native.output0[i] = (half)value;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_f32(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        float value = asfloat(pc.value_bits_lo);
        gFill_f32.output0[i] = value;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void fill_inplace_f64(uint3 tid : SV_DispatchThreadID, uniform PushConsts pc) {
    uint i = tid.x;
    if (i < pc.len) {
        double value = (double)asfloat(pc.value_bits_lo);
        if (pc.use_f64_bits != 0u) {
            value = asdouble(pc.value_bits_lo, pc.value_bits_hi);
        }
        gFill_f64.output0[i] = value;
    }
}
