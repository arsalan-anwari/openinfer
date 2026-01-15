use syn::parse::{ParseStream, Result};
use syn::Token;

use crate::kw;
use crate::types::InitValue;

mod init;
mod pattern;
mod ref_attr;

pub struct ParsedAttrs {
    pub init: Option<InitValue>,
    pub ref_name: Option<syn::LitStr>,
    pub pattern: Option<syn::LitStr>,
    pub table: bool,
    pub auto_dim: Vec<syn::Ident>,
    pub fixed: Vec<(syn::Ident, syn::LitInt)>,
}

pub fn parse_attrs(input: ParseStream) -> Result<ParsedAttrs> {
    let mut init = None;
    let mut ref_name = None;
    let mut pattern = None;
    let mut table = false;
    let mut auto_dim = Vec::new();
    let mut fixed = Vec::new();
    while input.peek(Token![@]) {
        input.parse::<Token![@]>()?;
        if input.peek(kw::init) {
            if init.is_some() {
                return Err(input.error("duplicate @init attribute"));
            }
            input.parse::<kw::init>()?;
            init = Some(init::parse_init_value(input)?);
        } else if input.peek(Token![ref]) {
            if ref_name.is_some() {
                return Err(input.error("duplicate @ref attribute"));
            }
            input.parse::<Token![ref]>()?;
            ref_name = Some(ref_attr::parse_ref_name(input)?);
        } else if input.peek(kw::pattern) {
            if pattern.is_some() {
                return Err(input.error("duplicate @pattern attribute"));
            }
            input.parse::<kw::pattern>()?;
            pattern = Some(pattern::parse_pattern(input)?);
        } else if input.peek(kw::table) {
            if table {
                return Err(input.error("duplicate @table attribute"));
            }
            input.parse::<kw::table>()?;
            table = true;
        } else if input.peek(kw::auto_dim) {
            if !auto_dim.is_empty() {
                return Err(input.error("duplicate @auto_dim attribute"));
            }
            input.parse::<kw::auto_dim>()?;
            let content;
            syn::parenthesized!(content in input);
            while !content.is_empty() {
                auto_dim.push(content.parse()?);
                if content.peek(Token![,]) {
                    content.parse::<Token![,]>()?;
                }
            }
            if auto_dim.is_empty() {
                return Err(input.error("@auto_dim requires at least one index"));
            }
        } else if input.peek(kw::fixed) {
            if !fixed.is_empty() {
                return Err(input.error("duplicate @fixed attribute"));
            }
            input.parse::<kw::fixed>()?;
            let content;
            syn::parenthesized!(content in input);
            while !content.is_empty() {
                let ident: syn::Ident = content.parse()?;
                content.parse::<Token![=]>()?;
                let value: syn::LitInt = content.parse()?;
                fixed.push((ident, value));
                if content.peek(Token![,]) {
                    content.parse::<Token![,]>()?;
                }
            }
            if fixed.is_empty() {
                return Err(input.error("@fixed requires at least one entry"));
            }
        } else {
            return Err(input.error("unsupported attribute"));
        }
    }
    Ok(ParsedAttrs {
        init,
        ref_name,
        pattern,
        table,
        auto_dim,
        fixed,
    })
}
