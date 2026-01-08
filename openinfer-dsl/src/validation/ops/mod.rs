use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::{OpAttrValue, OpSetting};

mod relu;

type OpAttrHandler = fn(&Ident, &[OpSetting]) -> syn::Result<TokenStream>;

pub(crate) fn op_attrs_expr(op: &Ident, settings: &[OpSetting]) -> syn::Result<TokenStream> {
    let op_name = op.to_string();
    if let Some(handler) = handler_for(&op_name) {
        return handler(op, settings);
    }

    if settings.is_empty() {
        Ok(quote! { ::openinfer::OpAttrs::None })
    } else {
        let setting = settings
            .first()
            .map(|s| s.name.span())
            .unwrap_or_else(|| op.span());
        Err(syn::Error::new(setting, "op does not support settings"))
    }
}

fn handler_for(name: &str) -> Option<OpAttrHandler> {
    const OPS: &[(&str, OpAttrHandler)] = &[("relu", relu::build_attrs)];
    OPS.iter()
        .find_map(|(op_name, handler)| (*op_name == name).then_some(*handler))
}

struct SettingsMap {
    op: String,
    settings: HashMap<String, OpSetting>,
}

impl SettingsMap {
    fn new(op: &Ident, settings: &[OpSetting]) -> syn::Result<Self> {
        let mut map = HashMap::new();
        for setting in settings {
            let key = setting.name.to_string();
            if map.contains_key(&key) {
                return Err(syn::Error::new(
                    setting.name.span(),
                    format!("duplicate {} setting: {}", op, key),
                ));
            }
            map.insert(key, setting.clone());
        }
        Ok(Self {
            op: op.to_string(),
            settings: map,
        })
    }

    fn take_value(&mut self, key: &str) -> Option<OpAttrValue> {
        self.settings.remove(key).map(|setting| setting.value)
    }

    fn ensure_empty(self) -> syn::Result<()> {
        if let Some((key, setting)) = self.settings.into_iter().next() {
            return Err(syn::Error::new(
                setting.name.span(),
                format!("unsupported {} setting: {}", self.op, key),
            ));
        }
        Ok(())
    }
}

fn attr_value_expr(value: &OpAttrValue) -> TokenStream {
    match value {
        OpAttrValue::Literal(val) => {
            if val.is_infinite() {
                if val.is_sign_negative() {
                    quote! { ::openinfer::AttrValue::Literal(::std::f32::NEG_INFINITY) }
                } else {
                    quote! { ::openinfer::AttrValue::Literal(::std::f32::INFINITY) }
                }
            } else {
                let lit = proc_macro2::Literal::f32_unsuffixed(*val);
                quote! { ::openinfer::AttrValue::Literal(#lit) }
            }
        }
        OpAttrValue::Var(ident) => {
            let s = ident.to_string();
            quote! { ::openinfer::AttrValue::Var(#s.to_string()) }
        }
    }
}
