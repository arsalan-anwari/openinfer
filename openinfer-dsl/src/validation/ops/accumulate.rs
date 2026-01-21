use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::codegen::memory::match_dtype;
use crate::types::{OpAttrValue, OpSetting};

use super::SettingsMap;

pub(crate) fn build_attrs(op: &Ident, settings: &[OpSetting]) -> syn::Result<TokenStream> {
    let mut settings = SettingsMap::new(op, settings)?;
    let acc_value = settings.take_value("acc");

    settings.ensure_empty()?;

    let Some(acc_value) = acc_value else {
        return Ok(quote! { ::openinfer::OpAttrs::None });
    };

    let acc_ident = match acc_value {
        OpAttrValue::Var(ident) => ident,
        _ => {
            return Err(syn::Error::new(
                op.span(),
                "acc setting expects a dtype identifier like i32",
            ))
        }
    };
    let acc_dtype = match_dtype(&acc_ident)?;

    Ok(quote! {
        ::openinfer::OpAttrs::Accumulate {
            dtype: #acc_dtype,
        }
    })
}
