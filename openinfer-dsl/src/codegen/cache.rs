use quote::quote;

use crate::types::{CacheAccess, CacheIndexExpr, CacheIndexValue};

pub(crate) fn cache_access_expr(access: &CacheAccess) -> syn::Result<proc_macro2::TokenStream> {
    let base = access.name.to_string();
    let bracketed = access.bracketed;
    let indices = access.indices.iter().map(cache_index_expr);
    Ok(quote! {
        ::openinfer::CacheAccess {
            base: #base.to_string(),
            indices: vec![#(#indices),*],
            bracketed: #bracketed,
        }
    })
}

fn cache_index_expr(index: &CacheIndexExpr) -> proc_macro2::TokenStream {
    match index {
        CacheIndexExpr::Single(value) => {
            let value = cache_index_value(value);
            quote! { ::openinfer::CacheIndexExpr::Single(#value) }
        }
        CacheIndexExpr::Slice { start, end } => {
            let start = cache_index_value_opt(start);
            let end = cache_index_value_opt(end);
            quote! {
                ::openinfer::CacheIndexExpr::Slice {
                    start: #start,
                    end: #end,
                }
            }
        }
    }
}

fn cache_index_value_opt(value: &Option<CacheIndexValue>) -> proc_macro2::TokenStream {
    match value {
        Some(value) => {
            let out = cache_index_value(value);
            quote! { Some(#out) }
        }
        None => quote! { None },
    }
}

fn cache_index_value(value: &CacheIndexValue) -> proc_macro2::TokenStream {
    match value {
        CacheIndexValue::Ident(ident) => {
            let name = ident.to_string();
            quote! { ::openinfer::CacheIndexValue::Ident(#name.to_string()) }
        }
        CacheIndexValue::Lit(value) => {
            quote! { ::openinfer::CacheIndexValue::Lit(#value) }
        }
    }
}
