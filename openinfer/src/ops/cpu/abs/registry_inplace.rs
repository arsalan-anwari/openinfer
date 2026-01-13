use crate::graph::OpAttrs;
use crate::tensor::DType;

#[allow(dead_code)]
pub fn supports_abs_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::I8, [DType::I8], OpAttrs::None)
            | (DType::I16, [DType::I16], OpAttrs::None)
            | (DType::F32, [DType::F32], OpAttrs::None)
            | (DType::F64, [DType::F64], OpAttrs::None)
            | (DType::I32, [DType::I32], OpAttrs::None)
            | (DType::I64, [DType::I64], OpAttrs::None)
            | (DType::U8, [DType::U8], OpAttrs::None)
            | (DType::U16, [DType::U16], OpAttrs::None)
            | (DType::U32, [DType::U32], OpAttrs::None)
            | (DType::U64, [DType::U64], OpAttrs::None)
            | (DType::Bool, [DType::Bool], OpAttrs::None)
    )
}
