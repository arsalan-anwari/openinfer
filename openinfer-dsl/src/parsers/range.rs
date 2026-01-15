use syn::parse::{ParseStream, Result};
use syn::{Ident, LitInt};

use crate::types::RangeValue;

pub(crate) fn parse_range_value(input: ParseStream) -> Result<RangeValue> {
    if input.peek(LitInt) {
        Ok(RangeValue::Lit(input.parse()?))
    } else if input.peek(Ident) {
        Ok(RangeValue::Ident(input.parse()?))
    } else {
        Err(input.error("expected identifier or integer for loop range"))
    }
}
