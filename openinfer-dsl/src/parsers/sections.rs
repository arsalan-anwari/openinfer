use syn::parse::{Parse, ParseStream, Result};
use syn::{braced, parenthesized, Ident, Token};

use crate::attributes;
use crate::kw;
use crate::parsers::dims::parse_dims;
use crate::types::{BlockSection, GraphDsl, MemoryKindToken, MemorySection, Section, VarDecl};

impl Parse for GraphDsl {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut sections = Vec::new();
        while !input.is_empty() {
            if input.peek(kw::dynamic)
                || input.peek(kw::volatile)
                || input.peek(kw::constant)
                || input.peek(kw::persistent)
            {
                sections.push(Section::Memory(input.parse()?));
            } else if input.peek(kw::block) {
                sections.push(Section::Block(input.parse()?));
            } else {
                return Err(input.error("expected memory section or block"));
            }
        }
        Ok(Self { sections })
    }
}

impl Parse for MemorySection {
    fn parse(input: ParseStream) -> Result<Self> {
        let kind = if input.peek(kw::dynamic) {
            input.parse::<kw::dynamic>()?;
            MemoryKindToken::Dynamic
        } else if input.peek(kw::volatile) {
            input.parse::<kw::volatile>()?;
            MemoryKindToken::Volatile
        } else if input.peek(kw::constant) {
            input.parse::<kw::constant>()?;
            MemoryKindToken::Constant
        } else {
            input.parse::<kw::persistent>()?;
            MemoryKindToken::Persistent
        };

        let content;
        braced!(content in input);
        let mut vars = Vec::new();
        while !content.is_empty() {
            vars.push(content.parse()?);
        }

        Ok(Self { kind, vars })
    }
}

impl Parse for VarDecl {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        let mut table_indices = Vec::new();
        if input.peek(syn::token::Paren) {
            let content;
            parenthesized!(content in input);
            while !content.is_empty() {
                let ident: Ident = content.parse()?;
                table_indices.push(ident);
                if content.peek(Token![,]) {
                    content.parse::<Token![,]>()?;
                }
            }
            if table_indices.is_empty() {
                return Err(content.error("prefix table must declare at least one index"));
            }
        }
        input.parse::<Token![:]>()?;
        let dtype: Ident = input.parse()?;
        let dims = parse_dims(input)?;
        let attrs = attributes::parse_attrs(input)?;
        input.parse::<Token![;]>()?;
        Ok(Self {
            name,
            dtype,
            dims,
            init: attrs.init,
            ref_name: attrs.ref_name,
            pattern: attrs.pattern,
            table_indices,
            table: attrs.table,
            auto_dim: attrs.auto_dim,
            fixed: attrs.fixed,
        })
    }
}

impl Parse for BlockSection {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse::<kw::block>()?;
        let name: Ident = input.parse()?;
        let content;
        braced!(content in input);
        let mut nodes = Vec::new();
        while !content.is_empty() {
            nodes.push(content.parse()?);
        }
        Ok(Self { name, nodes })
    }
}
