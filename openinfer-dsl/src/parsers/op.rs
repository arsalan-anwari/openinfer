use syn::parse::{ParseStream, Result};
use syn::{Ident, LitInt, LitStr, Token};

use crate::parsers::var::parse_indices;
use crate::types::{OpArg, OpAttrValue, OpSetting, VarRef};

pub(crate) fn parse_op_arg(input: ParseStream) -> Result<OpArg> {
    let name: Ident = input.parse()?;
    if input.peek(Token![=]) {
        input.parse::<Token![=]>()?;
        let value = parse_op_attr_value(input)?;
        Ok(OpArg::Setting(OpSetting { name, value }))
    } else if input.peek(syn::token::Bracket) {
        let indices = parse_indices(input)?;
        Ok(OpArg::Input(VarRef { name, indices }))
    } else {
        Ok(OpArg::Input(VarRef {
            name,
            indices: Vec::new(),
        }))
    }
}

pub(crate) fn parse_op_attr_value(input: ParseStream) -> Result<OpAttrValue> {
    if input.peek(syn::token::Bracket) {
        let content;
        syn::bracketed!(content in input);
        let mut values = Vec::new();
        while !content.is_empty() {
            let negative = if content.peek(Token![-]) {
                content.parse::<Token![-]>()?;
                true
            } else {
                false
            };
            let lit: LitInt = content.parse()?;
            let mut value: i64 = lit.base10_parse()?;
            if negative {
                value = -value;
            }
            values.push(value);
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }
        return Ok(OpAttrValue::IntList(values));
    }
    let negative = if input.peek(Token![-]) {
        input.parse::<Token![-]>()?;
        true
    } else {
        false
    };

    if input.peek(syn::LitFloat) {
        let lit: syn::LitFloat = input.parse()?;
        let mut value: f64 = lit.base10_parse()?;
        if negative {
            value = -value;
        }
        return Ok(OpAttrValue::Double(value));
    }
    if input.peek(syn::LitBool) {
        let lit: syn::LitBool = input.parse()?;
        if negative {
            return Err(input.error("unexpected '-' before bool literal"));
        }
        return Ok(OpAttrValue::Bool(lit.value));
    }
    if input.peek(LitInt) {
        let lit: LitInt = input.parse()?;
        let mut value: i64 = lit.base10_parse()?;
        if negative {
            value = -value;
        }
        return Ok(OpAttrValue::Int(value));
    }
    if input.peek(LitStr) {
        let lit: LitStr = input.parse()?;
        if negative {
            return Err(input.error("unexpected '-' before string literal"));
        }
        return Ok(OpAttrValue::String(lit.value()));
    }
    if input.peek(Ident) {
        let ident: Ident = input.parse()?;
        if negative {
            return Err(input.error("unexpected '-' before identifier"));
        }
        let name = ident.to_string();
        if name == "true" {
            return Ok(OpAttrValue::Bool(true));
        }
        if name == "false" {
            return Ok(OpAttrValue::Bool(false));
        }
        if name == "inf" {
            return Ok(OpAttrValue::Double(f64::INFINITY));
        }
        return Ok(OpAttrValue::Var(ident));
    }

    Err(input.error("expected literal or identifier for op setting"))
}
