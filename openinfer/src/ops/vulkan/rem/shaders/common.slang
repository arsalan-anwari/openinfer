#ifndef OPENINFER_REM_COMMON_SLANG
#define OPENINFER_REM_COMMON_SLANG 1

struct RemPush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
};

#endif // OPENINFER_REM_COMMON_SLANG
