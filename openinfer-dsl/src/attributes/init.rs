use syn::parse::{ParseStream, Result};
use syn::{parenthesized, LitInt, Token};

use crate::types::InitValue;

pub fn parse_init_value(input: ParseStream) -> Result<InitValue> {
    let content;
    parenthesized!(content in input);
    let negative = if content.peek(Token![-]) {
        content.parse::<Token![-]>()?;
        true
    } else {
        false
    };
    if content.peek(syn::LitFloat) {
        let lit: syn::LitFloat = content.parse()?;
        Ok(InitValue::Float { lit, negative })
    } else if content.peek(LitInt) {
        let lit: LitInt = content.parse()?;
        Ok(InitValue::Int { lit, negative })
    } else {
        Err(content.error("expected numeric literal for init"))
    }
}
