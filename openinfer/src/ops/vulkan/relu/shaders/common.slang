#ifndef OPENINFER_RELU_COMMON_SLANG
#define OPENINFER_RELU_COMMON_SLANG 1

struct ReluPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
    float alpha_f32;
    float clamp_max_f32;
    int alpha_i32;
    int clamp_max_i32;
    uint alpha_i64_lo;
    uint alpha_i64_hi;
    uint clamp_i64_lo;
    uint clamp_i64_hi;
    uint alpha_f64_lo;
    uint alpha_f64_hi;
    uint clamp_f64_lo;
    uint clamp_f64_hi;
};

#endif // OPENINFER_RELU_COMMON_SLANG
