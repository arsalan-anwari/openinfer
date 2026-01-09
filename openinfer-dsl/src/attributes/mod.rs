use syn::parse::{ParseStream, Result};
use syn::Token;

use crate::{kw, InitValue};

mod init;
mod pattern;
mod ref_attr;

pub struct ParsedAttrs {
    pub init: Option<InitValue>,
    pub ref_name: Option<syn::LitStr>,
    pub pattern: Option<syn::LitStr>,
}

pub fn parse_attrs(input: ParseStream) -> Result<ParsedAttrs> {
    let mut init = None;
    let mut ref_name = None;
    let mut pattern = None;
    while input.peek(Token![@]) {
        input.parse::<Token![@]>()?;
        if input.peek(kw::init) {
            if init.is_some() {
                return Err(input.error("duplicate @init attribute"));
            }
            input.parse::<kw::init>()?;
            init = Some(init::parse_init_value(input)?);
        } else if input.peek(kw::reference) {
            if ref_name.is_some() {
                return Err(input.error("duplicate @reference attribute"));
            }
            input.parse::<kw::reference>()?;
            ref_name = Some(ref_attr::parse_ref_name(input)?);
        } else if input.peek(kw::pattern) {
            if pattern.is_some() {
                return Err(input.error("duplicate @pattern attribute"));
            }
            input.parse::<kw::pattern>()?;
            pattern = Some(pattern::parse_pattern(input)?);
        } else {
            return Err(input.error("unsupported attribute"));
        }
    }
    Ok(ParsedAttrs {
        init,
        ref_name,
        pattern,
    })
}
