#ifndef OPENINFER_FLOOR_DIV_COMMON_SLANG
#define OPENINFER_FLOOR_DIV_COMMON_SLANG 1

struct FloorDivPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
    uint mask_u64_lo;
    uint mask_u64_hi;
    uint mask_i64_lo;
    uint mask_i64_hi;
    uint mask_f32_bits;
    uint mask_f64_lo;
    uint mask_f64_hi;
    uint mask_u8;
    uint mask_bool;
};

#endif // OPENINFER_FLOOR_DIV_COMMON_SLANG
