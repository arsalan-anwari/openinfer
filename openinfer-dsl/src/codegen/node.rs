use quote::quote;
use crate::codegen::cache::cache_access_expr;
use crate::codegen::dims::dims_expr;
use crate::codegen::memory::match_dtype;
use crate::types::{Node, RangeValue, VarRef};
use crate::validation;

use crate::types::{AssignNode, AwaitNode, BranchNode, DepNode, LoopNode, OpNode, TransferNode, YieldNode};

pub(crate) fn node_stmt(node: &Node, block_name: &str) -> syn::Result<proc_macro2::TokenStream> {
    match node {
        Node::Loop(loop_node) => {
            let name = loop_node.name.to_string();
            let index = loop_node.index.to_string();
            let start = range_value_string(&loop_node.start);
            let end = range_value_string(&loop_node.end);
            let body_expr = loop_body_expr(&loop_node.body)?;
            Ok(quote! {
                let loop_body = #body_expr;
                let loop_node = g.make_loop_node(
                    #name.to_string(),
                    #index.to_string(),
                    #start.to_string(),
                    #end.to_string(),
                    loop_body,
                );
                g.add_prebuilt_node(#block_name, loop_node)?;
            })
        }
        _ => {
            let node_expr = node_kind_expr(node)?;
            Ok(quote! {
                g.add_node(#block_name, #node_expr)?;
            })
        }
    }
}

pub(crate) fn node_kind_expr(node: &Node) -> syn::Result<proc_macro2::TokenStream> {
    match node {
        Node::Assign(assign) => assign_node_expr(assign),
        Node::Op(op) => op_node_expr(op),
        Node::Branch(branch) => branch_node_expr(branch),
        Node::Barrier => Ok(quote! { ::openinfer::NodeKind::Barrier }),
        Node::Dep(node) => dep_node_expr(node),
        Node::CacheRead(node) => {
            let src = cache_access_expr(&node.src)?;
            let dst = var_ref_string(&node.dst);
            Ok(quote! {
                ::openinfer::NodeKind::CacheRead {
                    src: #src,
                    dst: #dst.to_string(),
                }
            })
        }
        Node::CacheWrite(node) => {
            let src = var_ref_string(&node.src);
            let dst = cache_access_expr(&node.dst)?;
            Ok(quote! {
                ::openinfer::NodeKind::CacheWrite {
                    src: #src.to_string(),
                    dst: #dst,
                }
            })
        }
        Node::CacheInc(node) => {
            let target = node.target.to_string();
            let amount = node.amount;
            Ok(quote! {
                ::openinfer::NodeKind::CacheIncrement {
                    target: #target.to_string(),
                    amount: #amount,
                }
            })
        }
        Node::CacheDec(node) => {
            let target = node.target.to_string();
            let amount = node.amount;
            Ok(quote! {
                ::openinfer::NodeKind::CacheDecrement {
                    target: #target.to_string(),
                    amount: #amount,
                }
            })
        }
        Node::CacheReset(node) => {
            let target = cache_access_expr(&node.target)?;
            Ok(quote! {
                ::openinfer::NodeKind::CacheReset {
                    target: #target,
                }
            })
        }
        Node::Transfer(node) => transfer_node_expr(node),
        Node::Loop(loop_node) => loop_node_expr(loop_node),
        Node::Yield(node) => yield_node_expr(node),
        Node::Await(node) => await_node_expr(node),
        Node::Return => Ok(quote! { ::openinfer::NodeKind::Return }),
    }
}

fn assign_node_expr(assign: &AssignNode) -> syn::Result<proc_macro2::TokenStream> {
    let name = assign.name.to_string();
    let dtype = match_dtype(&assign.dtype)?;
    let dims = dims_expr(&assign.dims);
    Ok(quote! {
        ::openinfer::NodeKind::Assign {
            name: #name.to_string(),
            dtype: #dtype,
            dims: #dims,
        }
    })
}

fn op_node_expr(op: &OpNode) -> syn::Result<proc_macro2::TokenStream> {
    let op_name = op.name.to_string();
    let op_kind = quote! {
        ::openinfer::OpKind::from_name(#op_name)
            .expect("unknown op name")
    };
    let inputs = op.inputs.iter().map(|i| {
        let s = var_ref_string(i);
        quote! { #s.to_string() }
    });
    let output = op.output.to_string();
    let attrs = validation::ops::op_attrs_expr(&op.name, &op.settings)?;
    Ok(quote! {
        ::openinfer::NodeKind::Op {
            op: #op_kind,
            attrs: #attrs,
            inputs: vec![#(#inputs),*],
            output: #output.to_string(),
        }
    })
}

fn loop_node_expr(loop_node: &LoopNode) -> syn::Result<proc_macro2::TokenStream> {
    let name = loop_node.name.to_string();
    let index = loop_node.index.to_string();
    let start = range_value_string(&loop_node.start);
    let end = range_value_string(&loop_node.end);
    let body_expr = loop_body_expr(&loop_node.body)?;
    Ok(quote! {
        ::openinfer::NodeKind::Loop {
            name: #name.to_string(),
            index: #index.to_string(),
            start: #start.to_string(),
            end: #end.to_string(),
            body: #body_expr,
        }
    })
}

fn yield_node_expr(node: &YieldNode) -> syn::Result<proc_macro2::TokenStream> {
    let vars = node.vars.iter().map(|var| {
        let name = var.to_string();
        quote! { #name.to_string() }
    });
    Ok(quote! {
        ::openinfer::NodeKind::Yield {
            vars: vec![#(#vars),*],
        }
    })
}

fn await_node_expr(node: &AwaitNode) -> syn::Result<proc_macro2::TokenStream> {
    let vars = node.vars.iter().map(|var| {
        let name = var.to_string();
        quote! { #name.to_string() }
    });
    Ok(quote! {
        ::openinfer::NodeKind::Await {
            vars: vec![#(#vars),*],
        }
    })
}

pub(crate) fn loop_body_expr(nodes: &[Node]) -> syn::Result<proc_macro2::TokenStream> {
    let mut stmts = Vec::new();
    for node in nodes {
        let stmt = match node {
            Node::Loop(loop_node) => {
                let name = loop_node.name.to_string();
                let index = loop_node.index.to_string();
                let start = range_value_string(&loop_node.start);
                let end = range_value_string(&loop_node.end);
                let body_expr = loop_body_expr(&loop_node.body)?;
                quote! {
                    let loop_body = #body_expr;
                    body.push(g.make_loop_node(
                        #name.to_string(),
                        #index.to_string(),
                        #start.to_string(),
                        #end.to_string(),
                        loop_body,
                    ));
                }
            }
            _ => {
                let node_expr = node_kind_expr(node)?;
                quote! {
                    body.push(g.make_node(#node_expr));
                }
            }
        };
        stmts.push(stmt);
    }
    Ok(quote! {{
        let mut body = Vec::new();
        #(#stmts)*
        body
    }})
}

fn branch_node_expr(branch: &BranchNode) -> syn::Result<proc_macro2::TokenStream> {
    let cond = if let Some(cond) = branch.cond.as_ref() {
        let cond = cond.to_string();
        quote! { Some(#cond.to_string()) }
    } else {
        quote! { None }
    };
    let then_block = branch.then_block.to_string();
    let else_block = if let Some(else_block) = branch.else_block.as_ref() {
        let else_block = else_block.to_string();
        quote! { Some(#else_block.to_string()) }
    } else {
        quote! { None }
    };
    Ok(quote! {
        ::openinfer::NodeKind::Branch {
            cond: #cond,
            then_block: #then_block.to_string(),
            else_block: #else_block,
        }
    })
}

fn dep_node_expr(node: &DepNode) -> syn::Result<proc_macro2::TokenStream> {
    let after = node.after.to_string();
    let before = node.before.to_string();
    Ok(quote! {
        ::openinfer::NodeKind::Dep {
            after: #after.to_string(),
            before: #before.to_string(),
        }
    })
}

fn transfer_node_expr(node: &TransferNode) -> syn::Result<proc_macro2::TokenStream> {
    let src = var_ref_string(&node.src);
    let dst = var_ref_string(&node.dst);
    Ok(quote! {
        ::openinfer::NodeKind::Transfer {
            src: #src.to_string(),
            dst: #dst.to_string(),
        }
    })
}

fn range_value_string(value: &RangeValue) -> String {
    match value {
        RangeValue::Ident(ident) => ident.to_string(),
        RangeValue::Lit(lit) => lit.to_string(),
    }
}

fn var_ref_string(var_ref: &VarRef) -> String {
    let mut name = var_ref.name.to_string();
    if !var_ref.indices.is_empty() {
        let items = var_ref.indices.iter().map(|index| match index {
            crate::types::IndexExpr::Ident(ident) => ident.to_string(),
            crate::types::IndexExpr::Lit(lit) => lit.to_string(),
        });
        name.push('[');
        name.push_str(&items.collect::<Vec<_>>().join(","));
        name.push(']');
    }
    name
}
