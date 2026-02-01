#ifndef OPENINFER_IS_NAN_COMMON_SLANG
#define OPENINFER_IS_NAN_COMMON_SLANG 1

struct IsNanPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_IS_NAN_COMMON_SLANG
