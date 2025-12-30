use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::{braced, parenthesized, Ident, LitInt, Token};

mod kw {
    syn::custom_keyword!(dynamic);
    syn::custom_keyword!(volatile);
    syn::custom_keyword!(block);
    syn::custom_keyword!(assign);
    syn::custom_keyword!(op);
    syn::custom_keyword!(init);
}

#[proc_macro]
pub fn graph(input: TokenStream) -> TokenStream {
    let ast = syn::parse_macro_input!(input as GraphDsl);
    match ast.expand() {
        Ok(ts) => ts,
        Err(err) => err.to_compile_error().into(),
    }
}

struct GraphDsl {
    sections: Vec<Section>,
}

enum Section {
    Memory(MemorySection),
    Block(BlockSection),
}

struct MemorySection {
    kind: MemoryKindToken,
    vars: Vec<VarDecl>,
}

enum MemoryKindToken {
    Dynamic,
    Volatile,
}

struct VarDecl {
    name: Ident,
    dtype: Ident,
    dims: Vec<Dim>,
    init: Option<InitValue>,
}

enum Dim {
    Ident(Ident),
    Lit(LitInt),
}

enum InitValue {
    Float(syn::LitFloat),
    Int(LitInt),
}

struct BlockSection {
    name: Ident,
    nodes: Vec<Node>,
}

enum Node {
    Assign(AssignNode),
    Op(OpNode),
    Return,
}

struct AssignNode {
    name: Ident,
    dtype: Ident,
    dims: Vec<Dim>,
}

struct OpNode {
    name: Ident,
    inputs: Vec<Ident>,
    output: Ident,
}

impl Parse for GraphDsl {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut sections = Vec::new();
        while !input.is_empty() {
            if input.peek(kw::dynamic) || input.peek(kw::volatile) {
                sections.push(Section::Memory(input.parse()?));
            } else if input.peek(kw::block) {
                sections.push(Section::Block(input.parse()?));
            } else {
                return Err(input.error("expected memory section or block"));
            }
        }
        Ok(Self { sections })
    }
}

impl Parse for MemorySection {
    fn parse(input: ParseStream) -> Result<Self> {
        let kind = if input.peek(kw::dynamic) {
            input.parse::<kw::dynamic>()?;
            MemoryKindToken::Dynamic
        } else {
            input.parse::<kw::volatile>()?;
            MemoryKindToken::Volatile
        };

        let content;
        braced!(content in input);
        let mut vars = Vec::new();
        while !content.is_empty() {
            vars.push(content.parse()?);
        }

        Ok(Self { kind, vars })
    }
}

impl Parse for VarDecl {
    fn parse(input: ParseStream) -> Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let dtype: Ident = input.parse()?;
        let dims = parse_dims(input)?;
        let init = parse_init(input)?;
        input.parse::<Token![;]>()?;
        Ok(Self {
            name,
            dtype,
            dims,
            init,
        })
    }
}

impl Parse for BlockSection {
    fn parse(input: ParseStream) -> Result<Self> {
        input.parse::<kw::block>()?;
        let name: Ident = input.parse()?;
        let content;
        braced!(content in input);
        let mut nodes = Vec::new();
        while !content.is_empty() {
            nodes.push(content.parse()?);
        }
        Ok(Self { name, nodes })
    }
}

impl Parse for Node {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(kw::assign) {
            input.parse::<kw::assign>()?;
            let name: Ident = input.parse()?;
            input.parse::<Token![:]>()?;
            let dtype: Ident = input.parse()?;
            let dims = parse_dims(input)?;
            input.parse::<Token![;]>()?;
            Ok(Node::Assign(AssignNode { name, dtype, dims }))
        } else if input.peek(kw::op) {
            input.parse::<kw::op>()?;
            let name: Ident = input.parse()?;
            let content;
            parenthesized!(content in input);
            let mut inputs = Vec::new();
            while !content.is_empty() {
                let ident: Ident = content.parse()?;
                inputs.push(ident);
                if content.peek(Token![,]) {
                    content.parse::<Token![,]>()?;
                }
            }
            input.parse::<Token![>]>()?;
            input.parse::<Token![>]>()?;
            let output: Ident = input.parse()?;
            input.parse::<Token![;]>()?;
            Ok(Node::Op(OpNode { name, inputs, output }))
        } else if input.peek(Token![return]) {
            input.parse::<Token![return]>()?;
            input.parse::<Token![;]>()?;
            Ok(Node::Return)
        } else {
            Err(input.error("unsupported node"))
        }
    }
}

fn parse_dims(input: ParseStream) -> Result<Vec<Dim>> {
    let mut dims = Vec::new();
    if input.peek(syn::token::Bracket) {
        let content;
        syn::bracketed!(content in input);
        while !content.is_empty() {
            if content.peek(LitInt) {
                dims.push(Dim::Lit(content.parse()?));
            } else {
                dims.push(Dim::Ident(content.parse()?));
            }
            if content.peek(Token![,]) {
                content.parse::<Token![,]>()?;
            }
        }
    }
    Ok(dims)
}

fn parse_init(input: ParseStream) -> Result<Option<InitValue>> {
    if !input.peek(Token![@]) {
        return Ok(None);
    }
    input.parse::<Token![@]>()?;
    input.parse::<kw::init>()?;
    let content;
    parenthesized!(content in input);
    if content.peek(syn::LitFloat) {
        let lit: syn::LitFloat = content.parse()?;
        Ok(Some(InitValue::Float(lit)))
    } else if content.peek(LitInt) {
        let lit: LitInt = content.parse()?;
        Ok(Some(InitValue::Int(lit)))
    } else {
        Err(content.error("expected numeric literal for init"))
    }
}

impl GraphDsl {
    fn expand(self) -> Result<TokenStream> {
        let mut stmts = Vec::new();

        stmts.push(quote! { let mut g = ::openinfer::Graph::new(); });

        for section in self.sections {
            match section {
                Section::Memory(mem) => {
                    let kind_expr = match mem.kind {
                        MemoryKindToken::Dynamic => quote! { ::openinfer::MemoryKind::Dynamic },
                        MemoryKindToken::Volatile => quote! { ::openinfer::MemoryKind::Volatile },
                    };
                    for var in mem.vars {
                        let name = var.name.to_string();
                        let dtype = match_dtype(&var.dtype)?;
                        let dims = dims_expr(&var.dims);
                        let init = init_expr(&var.init, &var.dtype)?;
                        stmts.push(quote! {
                            g.add_var(#kind_expr, #name, #dtype, #dims, #init);
                        });
                    }
                }
                Section::Block(block) => {
                    let block_name = block.name.to_string();
                    stmts.push(quote! { g.add_block(#block_name); });
                    for node in block.nodes {
                        let node_expr = match node {
                            Node::Assign(assign) => {
                                let name = assign.name.to_string();
                                let dtype = match_dtype(&assign.dtype)?;
                                let dims = dims_expr(&assign.dims);
                                quote! {
                                    ::openinfer::NodeKind::Assign {
                                        name: #name.to_string(),
                                        dtype: #dtype,
                                        dims: #dims,
                                    }
                                }
                            }
                            Node::Op(op) => {
                                let kind = match_opkind(&op.name)?;
                                let inputs = op.inputs.iter().map(|i| {
                                    let s = i.to_string();
                                    quote! { #s.to_string() }
                                });
                                let output = op.output.to_string();
                                quote! {
                                    ::openinfer::NodeKind::Op {
                                        op: #kind,
                                        attrs: ::openinfer::OpAttrs::None,
                                        inputs: vec![#(#inputs),*],
                                        output: #output.to_string(),
                                    }
                                }
                            }
                            Node::Return => {
                                quote! { ::openinfer::NodeKind::Return }
                            }
                        };
                        stmts.push(quote! {
                            g.add_node(#block_name, #node_expr)?;
                        });
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

fn match_dtype(dtype: &Ident) -> Result<proc_macro2::TokenStream> {
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

fn match_opkind(op: &Ident) -> Result<proc_macro2::TokenStream> {
    let s = op.to_string();
    match s.as_str() {
        "add" => Ok(quote! { ::openinfer::OpKind::Add }),
        "mul" => Ok(quote! { ::openinfer::OpKind::Mul }),
        "abs" => Ok(quote! { ::openinfer::OpKind::Abs }),
        _ => Err(syn::Error::new(op.span(), "unsupported op")),
    }
}

fn dims_expr(dims: &[Dim]) -> proc_macro2::TokenStream {
    let items = dims.iter().map(|dim| match dim {
        Dim::Ident(ident) => {
            let s = ident.to_string();
            quote! { #s.to_string() }
        }
        Dim::Lit(lit) => {
            let s = lit.to_string();
            quote! { #s.to_string() }
        }
    });
    quote! { vec![#(#items),*] }
}

fn init_expr(init: &Option<InitValue>, dtype: &Ident) -> Result<proc_macro2::TokenStream> {
    let dtype_str = dtype.to_string();
    let out = match init {
        Some(InitValue::Float(lit)) => match dtype_str.as_str() {
            "f32" => quote! { Some(::openinfer::ScalarValue::F32(#lit as f32)) },
            "f64" => quote! { Some(::openinfer::ScalarValue::F64(#lit as f64)) },
            "i32" => quote! { Some(::openinfer::ScalarValue::I32(#lit as i32)) },
            "i64" => quote! { Some(::openinfer::ScalarValue::I64(#lit as i64)) },
            _ => {
                return Err(syn::Error::new(
                    dtype.span(),
                    "unsupported dtype for float init",
                ))
            }
        },
        Some(InitValue::Int(lit)) => match dtype_str.as_str() {
            "f32" => quote! { Some(::openinfer::ScalarValue::F32(#lit as f32)) },
            "f64" => quote! { Some(::openinfer::ScalarValue::F64(#lit as f64)) },
            "i32" => quote! { Some(::openinfer::ScalarValue::I32(#lit as i32)) },
            "i64" => quote! { Some(::openinfer::ScalarValue::I64(#lit as i64)) },
            "bool" => quote! { Some(::openinfer::ScalarValue::Bool(#lit != 0)) },
            _ => {
                return Err(syn::Error::new(
                    dtype.span(),
                    "unsupported dtype for int init",
                ))
            }
        },
        None => quote! { None },
    };
    Ok(out)
}
