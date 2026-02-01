#ifndef OPENINFER_LE_COMMON_SLANG
#define OPENINFER_LE_COMMON_SLANG 1

struct LePush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_LE_COMMON_SLANG
