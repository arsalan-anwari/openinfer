use crate::types::{Dim, DimAtom};
use quote::quote;

pub(crate) fn dims_expr(dims: &[Dim]) -> proc_macro2::TokenStream {
    let items = dims.iter().map(|dim| match dim {
        Dim::Ident(ident) => {
            let s = ident.to_string();
            quote! { #s.to_string() }
        }
        Dim::Lit(lit) => {
            let s = lit.to_string();
            quote! { #s.to_string() }
        }
        Dim::Mul { left, right } => {
            let left = dim_atom_string(left);
            let right = dim_atom_string(right);
            let s = format!("{}*{}", left, right);
            quote! { #s.to_string() }
        }
    });
    quote! { vec![#(#items),*] }
}

fn dim_atom_string(atom: &DimAtom) -> String {
    match atom {
        DimAtom::Ident(ident) => ident.to_string(),
        DimAtom::Lit(lit) => lit.to_string(),
    }
}
