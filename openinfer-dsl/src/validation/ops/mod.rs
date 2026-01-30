use std::collections::HashMap;

use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::codegen::memory::match_dtype;
use crate::types::{OpAttrValue, OpSetting};

pub(crate) fn op_attrs_expr(_op: &Ident, settings: &[OpSetting]) -> syn::Result<TokenStream> {
    let mut map = SettingsMap::new(settings)?;
    let mut items = Vec::new();
    for (name, setting) in map.settings.drain() {
        let name_literal = name.clone();
        let value_expr = attr_value_expr(&setting)?;
        items.push(quote! {
            ::openinfer::OpAttr {
                name: #name_literal.to_string(),
                value: #value_expr,
            }
        });
    }
    Ok(quote! {
        ::openinfer::OpAttrs { items: vec![#(#items),*] }
    })
}

struct SettingsMap {
    settings: HashMap<String, OpSetting>,
}

impl SettingsMap {
    fn new(settings: &[OpSetting]) -> syn::Result<Self> {
        let mut map = HashMap::new();
        for setting in settings {
            let key = setting.name.to_string();
            if map.contains_key(&key) {
                return Err(syn::Error::new(
                    setting.name.span(),
                    format!("duplicate setting: {}", key),
                ));
            }
            map.insert(key, setting.clone());
        }
        Ok(Self { settings: map })
    }
}

fn attr_value_expr(setting: &OpSetting) -> syn::Result<TokenStream> {
    if setting.name == "acc" || setting.name == "to" {
        match &setting.value {
            OpAttrValue::Var(ident) => {
                let dtype = match_dtype(ident)?;
                return Ok(quote! { ::openinfer::AttrValue::DType(#dtype) });
            }
            _ => {
                return Err(syn::Error::new(
                    setting.name.span(),
                    "acc must be a dtype identifier",
                ));
            }
        }
    }

    let value = &setting.value;
    Ok(match value {
        OpAttrValue::Float(val) => {
            if val.is_infinite() {
                if val.is_sign_negative() {
                    quote! { ::openinfer::AttrValue::Float(::std::f32::NEG_INFINITY) }
                } else {
                    quote! { ::openinfer::AttrValue::Float(::std::f32::INFINITY) }
                }
            } else {
                let lit = proc_macro2::Literal::f32_unsuffixed(*val);
                quote! { ::openinfer::AttrValue::Float(#lit) }
            }
        }
        OpAttrValue::Double(val) => {
            if val.is_infinite() {
                if val.is_sign_negative() {
                    quote! { ::openinfer::AttrValue::Double(::std::f64::NEG_INFINITY) }
                } else {
                    quote! { ::openinfer::AttrValue::Double(::std::f64::INFINITY) }
                }
            } else {
                let lit = proc_macro2::Literal::f64_unsuffixed(*val);
                quote! { ::openinfer::AttrValue::Double(#lit) }
            }
        }
        OpAttrValue::Int(val) => {
            let lit = proc_macro2::Literal::i64_unsuffixed(*val);
            quote! { ::openinfer::AttrValue::Int(#lit) }
        }
        OpAttrValue::Bool(val) => {
            quote! { ::openinfer::AttrValue::Bool(#val) }
        }
        OpAttrValue::String(val) => {
            quote! { ::openinfer::AttrValue::Str(#val.to_string()) }
        }
        OpAttrValue::IntList(values) => {
            quote! { ::openinfer::AttrValue::IntList(vec![#(#values),*]) }
        }
        OpAttrValue::Var(ident) => {
            let s = ident.to_string();
            quote! { ::openinfer::AttrValue::Var(#s.to_string()) }
        }
    })
}
