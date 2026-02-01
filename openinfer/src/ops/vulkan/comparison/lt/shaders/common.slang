#ifndef OPENINFER_LT_COMMON_SLANG
#define OPENINFER_LT_COMMON_SLANG 1

struct LtPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_LT_COMMON_SLANG
