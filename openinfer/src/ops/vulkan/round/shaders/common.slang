#ifndef OPENINFER_ROUND_COMMON_SLANG
#define OPENINFER_ROUND_COMMON_SLANG 1

struct RoundPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_ROUND_COMMON_SLANG
