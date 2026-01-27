use crate::registry::op_dtypes::{OpDTypeSupport, ACC_INT_PAIRS};
use crate::tensor::DType;

macro_rules! define_add_matchers {
    (
        normal: [$(($nvar:ident, $nty:ty)),+ $(,)?],
        packed_signed: [$(($psvar:ident, $psty:ty, $psw:expr)),+ $(,)?],
        packed_unsigned: [$(($puvar:ident, $puty:ty, $puw:expr)),+ $(,)?],
        acc_signed: [$($asvar:ident),+ $(,)?],
        acc_unsigned: [$($auvar:ident),+ $(,)?]
    ) => {
        pub const ADD_NORMAL_DTYPES: &[DType] = &[
            $(DType::$nvar,)+
            $(DType::$psvar,)+
            $(DType::$puvar,)+
        ];

        macro_rules! add_normal_match {
            ($a:expr, $b:expr, $out:expr) => {
                match ($a, $b, $out) {
                    $((TensorValue::$nvar(a), TensorValue::$nvar(b), TensorValue::$nvar(out)) => {
                        add_normal::<$nty>(a, b, out)
                    },)+
                    $((TensorValue::$psvar(a), TensorValue::$psvar(b), TensorValue::$psvar(out)) => {
                        add_packed_signed::<$psty>(a, b, out, $psw)
                    },)+
                    $((TensorValue::$puvar(a), TensorValue::$puvar(b), TensorValue::$puvar(out)) => {
                        add_packed_unsigned::<$puty>(a, b, out, $puw)
                    },)+
                    _ => Err(anyhow!("dtype mismatch")),
                }
            };
        }

        macro_rules! add_inplace_match {
            ($out:expr, $b:expr) => {
                match ($out, $b) {
                    $((TensorValue::$nvar(a), TensorValue::$nvar(b)) => add_inplace::<$nty>(a, b),)+
                    $((TensorValue::$psvar(a), TensorValue::$psvar(b)) => {
                        add_packed_signed_inplace::<$psty>(a, b, $psw)
                    },)+
                    $((TensorValue::$puvar(a), TensorValue::$puvar(b)) => {
                        add_packed_unsigned_inplace::<$puty>(a, b, $puw)
                    },)+
                    _ => Err(anyhow!("dtype mismatch")),
                }
            };
        }

        macro_rules! add_accumulate_match {
            ($a:expr, $b:expr, $out:expr) => {
                match ($a, $b) {
                    $((TensorValue::$asvar(a), TensorValue::$asvar(b)) => {
                        add_accumulate_signed_out(a, b, $out)
                    },)+
                    $((TensorValue::$auvar(a), TensorValue::$auvar(b)) => {
                        add_accumulate_unsigned_out(a, b, $out)
                    },)+
                    _ => Err(anyhow!("dtype mismatch")),
                }
            };
        }
    };
}

define_add_matchers!(
    normal: [
        (F8E5M2, crate::F8E5M2),
        (BF16, crate::BF16),
        (F16, crate::F16),
        (F32, f32),
        (F64, f64),
        (I8, i8),
        (I16, i16),
        (I32, i32),
        (I64, i64),
        (U8, u8),
        (U16, u16),
        (U32, u32),
        (U64, u64),
        (Bool, bool),
        (Bitset, crate::Bitset)
    ],
    packed_signed: [
        (I1, crate::I1, 1),
        (I2, crate::I2, 2),
        (I4, crate::I4, 4)
    ],
    packed_unsigned: [
        (U1, crate::U1, 1),
        (U2, crate::U2, 2),
        (U4, crate::U4, 4)
    ],
    acc_signed: [I1, I2, I4, I8, I16, I32, I64],
    acc_unsigned: [U1, U2, U4, U8, U16, U32, U64]
);

pub const ADD_DTYPE_SUPPORT: OpDTypeSupport = OpDTypeSupport {
    normal: ADD_NORMAL_DTYPES,
    accumulate: ACC_INT_PAIRS,
};
