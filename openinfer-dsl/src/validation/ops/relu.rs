use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::types::{OpAttrValue, OpSetting};

use super::{attr_value_expr, SettingsMap};

pub(crate) fn build_attrs(op: &Ident, settings: &[OpSetting]) -> syn::Result<TokenStream> {
    let mut settings = SettingsMap::new(op, settings)?;

    let negative_slope =
        settings
            .take_value("negative_slope")
            .unwrap_or_else(|| OpAttrValue::Float(0.0));
    let clamp_max =
        settings
            .take_value("clamp_max")
            .unwrap_or_else(|| OpAttrValue::Float(f32::INFINITY));

    settings.ensure_empty()?;

    let negative_slope_expr = attr_value_expr(&negative_slope);
    let clamp_max_expr = attr_value_expr(&clamp_max);

    Ok(quote! {
        ::openinfer::OpAttrs::Relu {
            negative_slope: #negative_slope_expr,
            clamp_max: #clamp_max_expr,
        }
    })
}
