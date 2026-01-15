use proc_macro::TokenStream;

mod attributes;
mod codegen;
mod parsers;
mod types;
mod validation;

mod kw {
    syn::custom_keyword!(dynamic);
    syn::custom_keyword!(volatile);
    syn::custom_keyword!(constant);
    syn::custom_keyword!(persistent);
    syn::custom_keyword!(block);
    syn::custom_keyword!(assign);
    syn::custom_keyword!(op);
    syn::custom_keyword!(cache);
    syn::custom_keyword!(read);
    syn::custom_keyword!(write);
    syn::custom_keyword!(increment);
    syn::custom_keyword!(decrement);
    syn::custom_keyword!(reset);
    syn::custom_keyword!(init);
    syn::custom_keyword!(pattern);
    syn::custom_keyword!(table);
    syn::custom_keyword!(fixed);
    syn::custom_keyword!(auto_dim);
}

use crate::types::GraphDsl;

#[proc_macro]
pub fn graph(input: TokenStream) -> TokenStream {
    let ast = syn::parse_macro_input!(input as GraphDsl);
    match ast.expand() {
        Ok(ts) => ts,
        Err(err) => err.to_compile_error().into(),
    }
}
