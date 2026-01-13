use crate::graph::OpAttrs;
use crate::tensor::DType;

#[allow(dead_code)]
pub fn supports_relu_inplace(output_dtype: DType, input_dtypes: &[DType], attrs: &OpAttrs) -> bool {
    matches!(
        (output_dtype, input_dtypes, attrs),
        (DType::F32, [DType::F32], OpAttrs::Relu { .. })
            | (DType::F64, [DType::F64], OpAttrs::Relu { .. })
            | (DType::I8, [DType::I8], OpAttrs::Relu { .. })
            | (DType::I16, [DType::I16], OpAttrs::Relu { .. })
            | (DType::I32, [DType::I32], OpAttrs::Relu { .. })
            | (DType::I64, [DType::I64], OpAttrs::Relu { .. })
            | (DType::U8, [DType::U8], OpAttrs::Relu { .. })
            | (DType::U16, [DType::U16], OpAttrs::Relu { .. })
            | (DType::U32, [DType::U32], OpAttrs::Relu { .. })
            | (DType::U64, [DType::U64], OpAttrs::Relu { .. })
            | (DType::Bool, [DType::Bool], OpAttrs::Relu { .. })
    )
}
