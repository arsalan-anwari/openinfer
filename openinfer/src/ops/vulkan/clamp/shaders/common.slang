#ifndef OPENINFER_CLAMP_COMMON_SLANG
#define OPENINFER_CLAMP_COMMON_SLANG 1

struct ClampPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
    uint min_u64_lo;
    uint min_u64_hi;
    uint min_i64_lo;
    uint min_i64_hi;
    uint min_f32_bits;
    uint min_f64_lo;
    uint min_f64_hi;
    uint min_u8;
    uint min_bool;
    uint max_u64_lo;
    uint max_u64_hi;
    uint max_i64_lo;
    uint max_i64_hi;
    uint max_f32_bits;
    uint max_f64_lo;
    uint max_f64_hi;
    uint max_u8;
    uint max_bool;
};

#endif // OPENINFER_CLAMP_COMMON_SLANG
