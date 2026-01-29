use anyhow::{anyhow, Result};

use crate::tensor::{I1, I2, I4, Tensor};

use super::common::{SignedAcc, SignedInput};

pub fn abs_accumulate_signed<In, Acc>(a: &Tensor<In>, out: &mut Tensor<Acc>) -> Result<()>
where
    In: SignedInput,
    Acc: SignedAcc,
{
    if out.shape() != a.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    for (out_slot, value) in out.data.iter_mut().zip(a.data.iter()) {
        let abs_value = value.to_i64().wrapping_abs();
        *out_slot = Acc::from_i64(abs_value);
    }
    Ok(())
}

macro_rules! signed_acc_fn {
    ($name:ident, $in:ty, $acc:ty) => {
        pub fn $name(a: &Tensor<$in>, out: &mut Tensor<$acc>) -> Result<()> {
            abs_accumulate_signed::<$in, $acc>(a, out)
        }
    };
}

signed_acc_fn!(abs_i1_accumulate_i8, I1, i8);
signed_acc_fn!(abs_i1_accumulate_i16, I1, i16);
signed_acc_fn!(abs_i1_accumulate_i32, I1, i32);
signed_acc_fn!(abs_i1_accumulate_i64, I1, i64);
signed_acc_fn!(abs_i2_accumulate_i8, I2, i8);
signed_acc_fn!(abs_i2_accumulate_i16, I2, i16);
signed_acc_fn!(abs_i2_accumulate_i32, I2, i32);
signed_acc_fn!(abs_i2_accumulate_i64, I2, i64);
signed_acc_fn!(abs_i4_accumulate_i8, I4, i8);
signed_acc_fn!(abs_i4_accumulate_i16, I4, i16);
signed_acc_fn!(abs_i4_accumulate_i32, I4, i32);
signed_acc_fn!(abs_i4_accumulate_i64, I4, i64);
signed_acc_fn!(abs_i8_accumulate_i16, i8, i16);
signed_acc_fn!(abs_i8_accumulate_i32, i8, i32);
signed_acc_fn!(abs_i8_accumulate_i64, i8, i64);
signed_acc_fn!(abs_i16_accumulate_i32, i16, i32);
signed_acc_fn!(abs_i16_accumulate_i64, i16, i64);
signed_acc_fn!(abs_i32_accumulate_i64, i32, i64);
