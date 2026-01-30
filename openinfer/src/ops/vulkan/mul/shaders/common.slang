#ifndef OPENINFER_MUL_COMMON_SLANG
#define OPENINFER_MUL_COMMON_SLANG 1

struct MulPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_MUL_COMMON_SLANG
