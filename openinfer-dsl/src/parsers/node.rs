use syn::parse::{Parse, ParseStream, Result};
use syn::{parenthesized, Token};

use crate::kw;
use crate::parsers::cache::{parse_cache_access, parse_cache_amount};
use crate::parsers::dims::parse_dims;
use crate::parsers::op::parse_op_arg;
use crate::parsers::range::parse_range_value;
use crate::parsers::var::parse_var_ref;
use crate::types::{
    AssignNode, AwaitNode, BranchNode, CacheDecNode, CacheIncNode, CacheReadNode, CacheResetNode,
    CacheWriteNode, DepNode, LoopNode, Node, OpArg, OpNode, TransferNode, YieldNode,
};

impl Parse for Node {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(kw::cache) {
            input.parse::<kw::cache>()?;
            input.parse::<Token![.]>()?;
            if input.peek(kw::read) {
                input.parse::<kw::read>()?;
                let src = parse_cache_access(input)?;
                input.parse::<Token![>>]>()?;
                let dst = parse_var_ref(input)?;
                input.parse::<Token![;]>()?;
                Ok(Node::CacheRead(CacheReadNode { src, dst }))
            } else if input.peek(kw::write) {
                input.parse::<kw::write>()?;
                let src = parse_var_ref(input)?;
                input.parse::<Token![>>]>()?;
                let dst = parse_cache_access(input)?;
                input.parse::<Token![;]>()?;
                Ok(Node::CacheWrite(CacheWriteNode { src, dst }))
            } else if input.peek(kw::increment) {
                input.parse::<kw::increment>()?;
                let amount = parse_cache_amount(input)?;
                let target = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Node::CacheInc(CacheIncNode { target, amount }))
            } else if input.peek(kw::decrement) {
                input.parse::<kw::decrement>()?;
                let amount = parse_cache_amount(input)?;
                let target = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Node::CacheDec(CacheDecNode { target, amount }))
            } else if input.peek(kw::reset) {
                input.parse::<kw::reset>()?;
                let target = parse_cache_access(input)?;
                input.parse::<Token![;]>()?;
                Ok(Node::CacheReset(CacheResetNode { target }))
            } else {
                Err(input.error("unsupported cache operation"))
            }
        } else if input.peek(kw::assign) {
            input.parse::<kw::assign>()?;
            let name = input.parse()?;
            input.parse::<Token![:]>()?;
            let dtype = input.parse()?;
            let dims = parse_dims(input)?;
            input.parse::<Token![;]>()?;
            Ok(Node::Assign(AssignNode { name, dtype, dims }))
        } else if input.peek(kw::op) {
            input.parse::<kw::op>()?;
            let name = input.parse()?;
            let content;
            parenthesized!(content in input);
            let mut inputs = Vec::new();
            let mut settings = Vec::new();
            let mut seen_setting = false;
            while !content.is_empty() {
                let arg = parse_op_arg(&content)?;
                match arg {
                    OpArg::Input(ident) => {
                        if seen_setting {
                            return Err(content.error("positional args must come before settings"));
                        }
                        inputs.push(ident);
                    }
                    OpArg::Setting(setting) => {
                        seen_setting = true;
                        settings.push(setting);
                    }
                }
                if content.peek(Token![,]) {
                    content.parse::<Token![,]>()?;
                }
            }
            input.parse::<Token![>]>()?;
            input.parse::<Token![>]>()?;
            let output = input.parse()?;
            input.parse::<Token![;]>()?;
            Ok(Node::Op(OpNode {
                name,
                inputs,
                settings,
                output,
            }))
        } else if input.peek(kw::branch) {
            input.parse::<kw::branch>()?;
            let first = input.parse()?;
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                Ok(Node::Branch(BranchNode {
                    cond: None,
                    then_block: first,
                    else_block: None,
                }))
            } else {
                let second = input.parse()?;
                if input.peek(Token![;]) {
                    return Err(input.error("branch expects condition and two target blocks"));
                }
                let third = input.parse()?;
                input.parse::<Token![;]>()?;
                Ok(Node::Branch(BranchNode {
                    cond: Some(first),
                    then_block: second,
                    else_block: Some(third),
                }))
            }
        } else if input.peek(kw::barrier) {
            input.parse::<kw::barrier>()?;
            input.parse::<Token![;]>()?;
            Ok(Node::Barrier)
        } else if input.peek(kw::dep) {
            input.parse::<kw::dep>()?;
            input.parse::<kw::after>()?;
            let after_content;
            parenthesized!(after_content in input);
            let after = after_content.parse()?;
            input.parse::<kw::before>()?;
            let before_content;
            parenthesized!(before_content in input);
            let before = before_content.parse()?;
            input.parse::<Token![;]>()?;
            Ok(Node::Dep(DepNode { after, before }))
        } else if input.peek(Token![loop]) {
            input.parse::<Token![loop]>()?;
            let name = input.parse()?;
            let content;
            parenthesized!(content in input);
            let index = content.parse()?;
            content.parse::<Token![in]>()?;
            let start = parse_range_value(&content)?;
            content.parse::<Token![..]>()?;
            let end = parse_range_value(&content)?;
            let body_content;
            syn::braced!(body_content in input);
            let mut body = Vec::new();
            while !body_content.is_empty() {
                body.push(body_content.parse()?);
            }
            Ok(Node::Loop(LoopNode {
                name,
                index,
                start,
                end,
                body,
            }))
        } else if input.peek(Token![yield]) {
            input.parse::<Token![yield]>()?;
            let mut vars = Vec::new();
            while !input.peek(Token![;]) {
                vars.push(input.parse()?);
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                } else {
                    break;
                }
            }
            if vars.is_empty() {
                return Err(input.error("yield expects at least one variable"));
            }
            input.parse::<Token![;]>()?;
            Ok(Node::Yield(YieldNode { vars }))
        } else if input.peek(Token![await]) {
            input.parse::<Token![await]>()?;
            let mut vars = Vec::new();
            while !input.peek(Token![;]) {
                vars.push(input.parse()?);
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                } else {
                    break;
                }
            }
            if vars.is_empty() {
                return Err(input.error("await expects at least one variable"));
            }
            input.parse::<Token![;]>()?;
            Ok(Node::Await(AwaitNode { vars }))
        } else if input.peek(kw::transfer) {
            input.parse::<kw::transfer>()?;
            let src = parse_var_ref(input)?;
            input.parse::<Token![>>]>()?;
            let dst = parse_var_ref(input)?;
            input.parse::<Token![;]>()?;
            Ok(Node::Transfer(TransferNode { src, dst }))
        } else if input.peek(Token![return]) {
            input.parse::<Token![return]>()?;
            input.parse::<Token![;]>()?;
            Ok(Node::Return)
        } else {
            Err(input.error("unsupported node"))
        }
    }
}
