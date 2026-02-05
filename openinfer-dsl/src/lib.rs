//! Procedural macro DSL for building OpenInfer graphs.
//!
//! The `graph!` macro parses a compact DSL into `openinfer::Graph` structures.
//! It is intended for ergonomics in tests and examples.
//!
//! ## DSL structure
//! - Memory sections: `dynamic`, `volatile`, `constant`, `persistent`
//! - Blocks: `block entry { ... }`
//! - Nodes: `assign`, `op`, `branch`, `loop`, `yield`, `await`, cache ops
//!
//! ## Expansion
//! The macro expands into Rust code that constructs `Graph` values at runtime.
//!
//! ## Example
//! ```no_run
//! use openinfer::graph;
//! let g = graph! {
//!     dynamic { x: f32[B]; }
//!     block entry {
//!         return;
//!     }
//! };
//! ```
use proc_macro::TokenStream;

mod attributes;
mod codegen;
mod parsers;
mod types;
mod validation;

mod kw {
    syn::custom_keyword!(dynamic);
    syn::custom_keyword!(volatile);
    syn::custom_keyword!(constant);
    syn::custom_keyword!(persistent);
    syn::custom_keyword!(block);
    syn::custom_keyword!(assign);
    syn::custom_keyword!(op);
    syn::custom_keyword!(branch);
    syn::custom_keyword!(barrier);
    syn::custom_keyword!(dep);
    syn::custom_keyword!(after);
    syn::custom_keyword!(before);
    syn::custom_keyword!(transfer);
    syn::custom_keyword!(cache);
    syn::custom_keyword!(read);
    syn::custom_keyword!(write);
    syn::custom_keyword!(increment);
    syn::custom_keyword!(decrement);
    syn::custom_keyword!(reset);
    syn::custom_keyword!(init);
    syn::custom_keyword!(pattern);
    syn::custom_keyword!(table);
    syn::custom_keyword!(fixed);
    syn::custom_keyword!(auto_dim);
}

use crate::types::GraphDsl;

/// Build an OpenInfer `Graph` from the DSL input.
#[proc_macro]
pub fn graph(input: TokenStream) -> TokenStream {
    let ast = syn::parse_macro_input!(input as GraphDsl);
    match ast.expand() {
        Ok(ts) => ts,
        Err(err) => err.to_compile_error().into(),
    }
}

#[cfg(test)]
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tests/openinfer-dsl/parse_tests.rs"
));
