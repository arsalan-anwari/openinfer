pub(crate) mod cache;
pub(crate) mod dims;
pub(crate) mod memory;
pub(crate) mod node;

use proc_macro::TokenStream;
use quote::quote;

use crate::codegen::dims::dims_expr;
use crate::codegen::memory::{init_expr, match_dtype};
use crate::codegen::node::node_stmt;
use crate::types::{GraphDsl, MemoryKindToken, Section};

impl GraphDsl {
    pub(crate) fn expand(self) -> syn::Result<TokenStream> {
        let mut stmts = Vec::new();

        stmts.push(quote! { let mut g = ::openinfer::Graph::new(); });

        for section in self.sections {
            match section {
                Section::Memory(mem) => {
                    let kind_expr = match mem.kind {
                        MemoryKindToken::Dynamic => quote! { ::openinfer::MemoryKind::Dynamic },
                        MemoryKindToken::Volatile => quote! { ::openinfer::MemoryKind::Volatile },
                        MemoryKindToken::Constant => quote! { ::openinfer::MemoryKind::Constant },
                        MemoryKindToken::Persistent => quote! { ::openinfer::MemoryKind::Persistent },
                    };
                    for var in mem.vars {
                        let name = var.name.to_string();
                        let dtype = match_dtype(&var.dtype)?;
                        let dims = dims_expr(&var.dims);
                        let init = init_expr(&var.init, &var.dtype)?;
                        let ref_name = match var.ref_name {
                            Some(lit) => quote! { Some(#lit.to_string()) },
                            None => quote! { None },
                        };
                        let pattern = match var.pattern {
                            Some(lit) => quote! { Some(#lit.to_string()) },
                            None => quote! { None },
                        };
                        let table = var.table;
                        let table_indices = var.table_indices.iter().map(|index| {
                            let s = index.to_string();
                            quote! { #s.to_string() }
                        });
                        let auto_dim = var.auto_dim.iter().map(|index| {
                            let s = index.to_string();
                            quote! { #s.to_string() }
                        });
                        let fixed_entries: Vec<proc_macro2::TokenStream> = var
                            .fixed
                            .iter()
                            .map(|(name, value)| {
                                let name = name.to_string();
                                let value: usize = value.base10_parse()?;
                                Ok(quote! { (#name.to_string(), #value) })
                            })
                            .collect::<syn::Result<Vec<_>>>()?;
                        stmts.push(quote! {
                            g.add_var(
                                #kind_expr,
                                #name,
                                #dtype,
                                #dims,
                                #init,
                                #ref_name,
                                vec![#(#table_indices),*],
                                #pattern,
                                #table,
                                vec![#(#auto_dim),*],
                                vec![#(#fixed_entries),*],
                            );
                        });
                    }
                }
                Section::Block(block) => {
                    let block_name = block.name.to_string();
                    stmts.push(quote! { g.add_block(#block_name); });
                    for node in block.nodes {
                        let node_stmt = node_stmt(&node, &block_name)?;
                        stmts.push(node_stmt);
                    }
                }
            }
        }

        let out = quote! {{
            #(#stmts)*
            g
        }};
        Ok(out.into())
    }
}
