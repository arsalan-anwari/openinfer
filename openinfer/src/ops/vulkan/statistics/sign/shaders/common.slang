#ifndef OPENINFER_SIGN_COMMON_SLANG
#define OPENINFER_SIGN_COMMON_SLANG 1

struct SignPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_SIGN_COMMON_SLANG
