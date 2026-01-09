use syn::ext::IdentExt;
use syn::parse::{ParseStream, Result};
use syn::{parenthesized, Ident};

pub fn parse_ref_name(input: ParseStream) -> Result<syn::LitStr> {
    let content;
    parenthesized!(content in input);
    if content.peek(syn::LitStr) {
        content.parse()
    } else {
        let ident = Ident::parse_any(&content)?;
        Ok(syn::LitStr::new(&ident.to_string(), ident.span()))
    }
}
