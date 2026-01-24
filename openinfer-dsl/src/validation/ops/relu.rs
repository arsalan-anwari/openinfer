use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::types::{OpAttrValue, OpSetting};

use super::{attr_value_expr, SettingsMap};

pub(crate) fn build_attrs(op: &Ident, settings: &[OpSetting]) -> syn::Result<TokenStream> {
    let mut settings = SettingsMap::new(op, settings)?;

    let alpha =
        settings
            .take_value("alpha")
            .unwrap_or_else(|| OpAttrValue::Double(0.0));
    let clamp_max =
        settings
            .take_value("clamp_max")
            .unwrap_or_else(|| OpAttrValue::Double(f64::INFINITY));

    settings.ensure_empty()?;

    let alpha_expr = attr_value_expr(&alpha);
    let clamp_max_expr = attr_value_expr(&clamp_max);

    Ok(quote! {
        ::openinfer::OpAttrs::Relu {
            alpha: #alpha_expr,
            clamp_max: #clamp_max_expr,
        }
    })
}
