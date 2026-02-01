#ifndef OPENINFER_IS_NEG_COMMON_SLANG
#define OPENINFER_IS_NEG_COMMON_SLANG 1

struct IsNegPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_IS_NEG_COMMON_SLANG
