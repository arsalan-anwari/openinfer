#ifndef OPENINFER_IS_FINITE_COMMON_SLANG
#define OPENINFER_IS_FINITE_COMMON_SLANG 1

struct IsFinitePush {
    uint len;
    uint tensor_count;
    uint params_offset;
    uint flags;
    uint input_len;
    uint _pad0;
    uint _pad1;
    uint _pad2;
};

#endif // OPENINFER_IS_FINITE_COMMON_SLANG
