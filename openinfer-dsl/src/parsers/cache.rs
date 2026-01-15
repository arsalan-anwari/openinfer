use syn::parse::{ParseStream, Result};
use syn::{Ident, LitInt, Token};

use crate::types::{CacheAccess, CacheIndexExpr, CacheIndexValue};

pub(crate) fn parse_cache_access(input: ParseStream) -> Result<CacheAccess> {
    let name: Ident = input.parse()?;
    if input.peek(syn::token::Bracket) {
        let indices = parse_cache_indices(input)?;
        Ok(CacheAccess {
            name,
            indices,
            bracketed: true,
        })
    } else {
        Ok(CacheAccess {
            name,
            indices: Vec::new(),
            bracketed: false,
        })
    }
}

fn parse_cache_indices(input: ParseStream) -> Result<Vec<CacheIndexExpr>> {
    let content;
    syn::bracketed!(content in input);
    let mut indices = Vec::new();
    while !content.is_empty() {
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
            indices.push(CacheIndexExpr::Slice {
                start: None,
                end: None,
            });
            continue;
        }
        let entry = if content.peek(Token![..]) {
            content.parse::<Token![..]>()?;
            let end = parse_cache_index_value_opt(&content)?;
            CacheIndexExpr::Slice { start: None, end }
        } else {
            let start = parse_cache_index_value(&content)?;
            if content.peek(Token![..]) {
                content.parse::<Token![..]>()?;
                let end = parse_cache_index_value_opt(&content)?;
                CacheIndexExpr::Slice {
                    start: Some(start),
                    end,
                }
            } else {
                CacheIndexExpr::Single(start)
            }
        };
        indices.push(entry);
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
            if content.is_empty() {
                indices.push(CacheIndexExpr::Slice {
                    start: None,
                    end: None,
                });
            }
        }
    }
    Ok(indices)
}

fn parse_cache_index_value_opt(input: ParseStream) -> Result<Option<CacheIndexValue>> {
    if input.is_empty() || input.peek(Token![,]) {
        return Ok(None);
    }
    Ok(Some(parse_cache_index_value(input)?))
}

fn parse_cache_index_value(input: ParseStream) -> Result<CacheIndexValue> {
    let negative = if input.peek(Token![-]) {
        input.parse::<Token![-]>()?;
        true
    } else {
        false
    };
    if input.peek(LitInt) {
        let lit: LitInt = input.parse()?;
        let mut value: i64 = lit.base10_parse()?;
        if negative {
            value = -value;
        }
        return Ok(CacheIndexValue::Lit(value));
    }
    if input.peek(Ident) {
        if negative {
            return Err(input.error("unexpected '-' before identifier"));
        }
        let ident: Ident = input.parse()?;
        return Ok(CacheIndexValue::Ident(ident));
    }
    Err(input.error("expected identifier or integer for cache index"))
}

pub(crate) fn parse_cache_amount(input: ParseStream) -> Result<i64> {
    if input.peek(LitInt) {
        let lit: LitInt = input.parse()?;
        let value: i64 = lit.base10_parse()?;
        return Ok(value);
    }
    Ok(1)
}
