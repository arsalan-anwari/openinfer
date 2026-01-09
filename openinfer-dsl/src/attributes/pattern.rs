use syn::parse::{ParseStream, Result};
use syn::{parenthesized, LitStr};

pub fn parse_pattern(input: ParseStream) -> Result<LitStr> {
    let content;
    parenthesized!(content in input);
    if content.peek(LitStr) {
        content.parse()
    } else {
        Err(content.error("expected string literal for pattern"))
    }
}
