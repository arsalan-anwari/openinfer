use crate::graph::OpAttrs;
use crate::tensor::DType;

pub fn supports_relu_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::F32, [DType::F32], OpAttrs::Relu { .. })
    )
}
