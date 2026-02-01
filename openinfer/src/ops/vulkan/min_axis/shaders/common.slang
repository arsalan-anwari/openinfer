#ifndef OPENINFER_MIN_AXIS_COMMON_SLANG
#define OPENINFER_MIN_AXIS_COMMON_SLANG 1

struct MinAxisPush {
    uint input_len;
    uint output_len;
    uint input_rank;
    uint output_rank;
    uint axes_mask;
    uint keepdims;
    uint _pad0;
    uint _pad1;
};

#endif // OPENINFER_MIN_AXIS_COMMON_SLANG
