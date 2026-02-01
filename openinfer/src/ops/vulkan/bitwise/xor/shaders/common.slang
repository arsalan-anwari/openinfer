#ifndef OPENINFER_XOR_COMMON_SLANG
#define OPENINFER_XOR_COMMON_SLANG 1

struct XorPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_XOR_COMMON_SLANG
