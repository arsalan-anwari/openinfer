use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

use crate::types::InitValue;

pub(crate) fn match_dtype(dtype: &Ident) -> syn::Result<TokenStream> {
    let s = dtype.to_string();
    match s.as_str() {
        "i8" => Ok(quote! { ::openinfer::DType::I8 }),
        "i16" => Ok(quote! { ::openinfer::DType::I16 }),
        "f32" => Ok(quote! { ::openinfer::DType::F32 }),
        "f64" => Ok(quote! { ::openinfer::DType::F64 }),
        "u8" => Ok(quote! { ::openinfer::DType::U8 }),
        "u16" => Ok(quote! { ::openinfer::DType::U16 }),
        "i32" => Ok(quote! { ::openinfer::DType::I32 }),
        "i64" => Ok(quote! { ::openinfer::DType::I64 }),
        "u32" => Ok(quote! { ::openinfer::DType::U32 }),
        "u64" => Ok(quote! { ::openinfer::DType::U64 }),
        "bool" => Ok(quote! { ::openinfer::DType::Bool }),
        "bitset" => Ok(quote! { ::openinfer::DType::Bitset }),
        "f16" => Ok(quote! { ::openinfer::DType::F16 }),
        _ => Err(syn::Error::new(dtype.span(), "unsupported dtype")),
    }
}

pub(crate) fn init_expr(init: &Option<InitValue>, dtype: &Ident) -> syn::Result<TokenStream> {
    let dtype_str = dtype.to_string();
    let out = match init {
        Some(InitValue::Float { lit, negative }) => {
            let lit_expr = if *negative {
                quote! { -#lit }
            } else {
                quote! { #lit }
            };
            match dtype_str.as_str() {
                "f16" => quote! {
                    Some(::openinfer::ScalarValue::F16(::openinfer::F16::from_f32(#lit_expr as f32)))
                },
                "f32" => quote! { Some(::openinfer::ScalarValue::F32(#lit_expr as f32)) },
                "f64" => quote! { Some(::openinfer::ScalarValue::F64(#lit_expr as f64)) },
                _ => {
                    return Err(syn::Error::new(
                        dtype.span(),
                        "float init requires f16/f32/f64 dtype",
                    ))
                }
            }
        }
        Some(InitValue::Int { lit, negative }) => {
            let lit_expr = if *negative {
                quote! { -#lit }
            } else {
                quote! { #lit }
            };
            let value: i128 = lit.base10_parse()?;
            let value = if *negative { -value } else { value };
            match dtype_str.as_str() {
                "i8" => {
                    if value < i8::MIN as i128 || value > i8::MAX as i128 {
                        return Err(syn::Error::new(dtype.span(), "i8 init out of range"));
                    }
                    quote! { Some(::openinfer::ScalarValue::I8(#lit_expr as i8)) }
                }
                "i16" => {
                    if value < i16::MIN as i128 || value > i16::MAX as i128 {
                        return Err(syn::Error::new(dtype.span(), "i16 init out of range"));
                    }
                    quote! { Some(::openinfer::ScalarValue::I16(#lit_expr as i16)) }
                }
                "i32" => {
                    if value < i32::MIN as i128 || value > i32::MAX as i128 {
                        return Err(syn::Error::new(dtype.span(), "i32 init out of range"));
                    }
                    quote! { Some(::openinfer::ScalarValue::I32(#lit_expr as i32)) }
                }
                "i64" => {
                    if value < i64::MIN as i128 || value > i64::MAX as i128 {
                        return Err(syn::Error::new(dtype.span(), "i64 init out of range"));
                    }
                    quote! { Some(::openinfer::ScalarValue::I64(#lit_expr as i64)) }
                }
                "u8" => {
                    if value < 0 || value > u8::MAX as i128 {
                        return Err(syn::Error::new(dtype.span(), "u8 init out of range"));
                    }
                    quote! { Some(::openinfer::ScalarValue::U8(#lit_expr as u8)) }
                }
                "u16" => {
                    if value < 0 || value > u16::MAX as i128 {
                        return Err(syn::Error::new(dtype.span(), "u16 init out of range"));
                    }
                    quote! { Some(::openinfer::ScalarValue::U16(#lit_expr as u16)) }
                }
                "u32" => {
                    if value < 0 || value > u32::MAX as i128 {
                        return Err(syn::Error::new(dtype.span(), "u32 init out of range"));
                    }
                    quote! { Some(::openinfer::ScalarValue::U32(#lit_expr as u32)) }
                }
                "u64" => {
                    if value < 0 || value > u64::MAX as i128 {
                        return Err(syn::Error::new(dtype.span(), "u64 init out of range"));
                    }
                    quote! { Some(::openinfer::ScalarValue::U64(#lit_expr as u64)) }
                }
                "bool" => {
                    if value != 0 && value != 1 {
                        return Err(syn::Error::new(dtype.span(), "bool init must be 0 or 1"));
                    }
                    quote! { Some(::openinfer::ScalarValue::Bool(#lit_expr != 0)) }
                }
                "bitset" => {
                    if value < 0 || value > u8::MAX as i128 {
                        return Err(syn::Error::new(dtype.span(), "bitset init out of range"));
                    }
                    quote! { Some(::openinfer::ScalarValue::Bitset(::openinfer::Bitset { bits: #lit_expr as u8 })) }
                }
                _ => {
                    return Err(syn::Error::new(
                        dtype.span(),
                        "integer init requires integer/bool/bitset dtype",
                    ))
                }
            }
        }
        None => quote! { None },
    };
    Ok(out)
}
