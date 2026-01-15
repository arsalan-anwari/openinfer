use syn::parse::{ParseStream, Result};
use syn::{Ident, LitInt, Token};

use crate::types::{Dim, DimAtom};

pub(crate) fn parse_dims(input: ParseStream) -> Result<Vec<Dim>> {
    let mut dims = Vec::new();
    if input.peek(syn::token::Bracket) {
        let content;
        syn::bracketed!(content in input);
        while !content.is_empty() {
            if content.peek(LitInt) {
                let lit: LitInt = content.parse()?;
                if content.peek(Token![*]) {
                    content.parse::<Token![*]>()?;
                    let right = parse_dim_atom(&content)?;
                    dims.push(Dim::Mul {
                        left: DimAtom::Lit(lit),
                        right,
                    });
                } else {
                    dims.push(Dim::Lit(lit));
                }
            } else {
                let ident: Ident = content.parse()?;
                if content.peek(Token![*]) {
                    content.parse::<Token![*]>()?;
                    let right = parse_dim_atom(&content)?;
                    dims.push(Dim::Mul {
                        left: DimAtom::Ident(ident),
                        right,
                    });
                } else {
                    dims.push(Dim::Ident(ident));
                }
            }
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }
    }
    Ok(dims)
}

fn parse_dim_atom(input: ParseStream) -> Result<DimAtom> {
    if input.peek(LitInt) {
        Ok(DimAtom::Lit(input.parse()?))
    } else if input.peek(Ident) {
        Ok(DimAtom::Ident(input.parse()?))
    } else {
        Err(input.error("expected identifier or integer for dimension expression"))
    }
}
