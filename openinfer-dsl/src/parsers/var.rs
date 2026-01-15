use syn::parse::{ParseStream, Result};
use syn::{Ident, LitInt, Token};

use crate::types::{IndexExpr, VarRef};

pub(crate) fn parse_indices(input: ParseStream) -> Result<Vec<IndexExpr>> {
    let content;
    syn::bracketed!(content in input);
    let mut indices = Vec::new();
    while !content.is_empty() {
        if content.peek(LitInt) {
            indices.push(IndexExpr::Lit(content.parse()?));
        } else {
            indices.push(IndexExpr::Ident(content.parse()?));
        }
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }
    if indices.is_empty() {
        return Err(content.error("prefix access must include at least one index"));
    }
    Ok(indices)
}

pub(crate) fn parse_var_ref(input: ParseStream) -> Result<VarRef> {
    let name: Ident = input.parse()?;
    let indices = if input.peek(syn::token::Bracket) {
        parse_indices(input)?
    } else {
        Vec::new()
    };
    Ok(VarRef { name, indices })
}
