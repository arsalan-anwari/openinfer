use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::types::{OpAttrValue, OpSetting};

use super::{attr_value_expr, SettingsMap};

pub(crate) fn build_attrs(op: &Ident, settings: &[OpSetting]) -> syn::Result<TokenStream> {
    let mut settings = SettingsMap::new(op, settings)?;

    let value = settings
        .take_value("value")
        .unwrap_or_else(|| OpAttrValue::Double(0.0));

    settings.ensure_empty()?;

    let value_expr = attr_value_expr(&value);

    Ok(quote! {
        ::openinfer::OpAttrs::Fill {
            value: #value_expr,
        }
    })
}
