#ifndef OPENINFER_ARGMAX_AXIS_COMMON_SLANG
#define OPENINFER_ARGMAX_AXIS_COMMON_SLANG 1

struct ArgmaxAxisPush {
    uint input_len;
    uint output_len;
    uint input_rank;
    uint output_rank;
    uint axes_mask;
    uint keepdims;
    uint axis;
    uint select_first;
};

#endif // OPENINFER_ARGMAX_AXIS_COMMON_SLANG
