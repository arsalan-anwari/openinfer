#ifndef OPENINFER_FILL_COMMON_SLANG
#define OPENINFER_FILL_COMMON_SLANG 1

struct FillPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
    uint value_u64_lo;
    uint value_u64_hi;
    uint value_i64_lo;
    uint value_i64_hi;
    uint value_f32_bits;
    uint value_f64_lo;
    uint value_f64_hi;
    uint value_u8;
    uint value_bool;
};

#endif // OPENINFER_FILL_COMMON_SLANG
