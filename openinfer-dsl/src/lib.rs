use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::{braced, parenthesized, Ident, LitInt, Token};

mod attributes;
mod kw {
    syn::custom_keyword!(dynamic);
    syn::custom_keyword!(volatile);
    syn::custom_keyword!(constant);
    syn::custom_keyword!(block);
    syn::custom_keyword!(assign);
    syn::custom_keyword!(op);
    syn::custom_keyword!(init);
    syn::custom_keyword!(reference);
    syn::custom_keyword!(pattern);
}

mod validation;

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
    Constant,
}

struct VarDecl {
    name: Ident,
    dtype: Ident,
    dims: Vec<Dim>,
    init: Option<InitValue>,
    ref_name: Option<syn::LitStr>,
    pattern: Option<syn::LitStr>,
    table_indices: Vec<Ident>,
}

enum Dim {
    Ident(Ident),
    Lit(LitInt),
}

enum InitValue {
    Float { lit: syn::LitFloat, negative: bool },
    Int { lit: LitInt, negative: bool },
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
    inputs: Vec<VarRef>,
    settings: Vec<OpSetting>,
    output: Ident,
}

#[derive(Clone)]
struct OpSetting {
    name: Ident,
    value: OpAttrValue,
}

#[derive(Clone)]
enum OpAttrValue {
    Literal(f32),
    Var(Ident),
}

enum OpArg {
    Input(VarRef),
    Setting(OpSetting),
}

struct VarRef {
    name: Ident,
    indices: Vec<IndexExpr>,
}

enum IndexExpr {
    Ident(Ident),
    Lit(LitInt),
}

impl Parse for GraphDsl {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut sections = Vec::new();
        while !input.is_empty() {
            if input.peek(kw::dynamic) || input.peek(kw::volatile) || input.peek(kw::constant) {
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
        } else if input.peek(kw::volatile) {
            input.parse::<kw::volatile>()?;
            MemoryKindToken::Volatile
        } else {
            input.parse::<kw::constant>()?;
            MemoryKindToken::Constant
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
        let mut table_indices = Vec::new();
        if input.peek(syn::token::Paren) {
            let content;
            parenthesized!(content in input);
            while !content.is_empty() {
                let ident: Ident = content.parse()?;
                table_indices.push(ident);
                if content.peek(Token![,]) {
                    content.parse::<Token![,]>()?;
                }
            }
            if table_indices.is_empty() {
                return Err(content.error("prefix table must declare at least one index"));
            }
        }
        input.parse::<Token![:]>()?;
        let dtype: Ident = input.parse()?;
        let dims = parse_dims(input)?;
        let attrs = attributes::parse_attrs(input)?;
        input.parse::<Token![;]>()?;
        Ok(Self {
            name,
            dtype,
            dims,
            init: attrs.init,
            ref_name: attrs.ref_name,
            pattern: attrs.pattern,
            table_indices,
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
            let output: Ident = input.parse()?;
            input.parse::<Token![;]>()?;
            Ok(Node::Op(OpNode {
                name,
                inputs,
                settings,
                output,
            }))
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


fn parse_op_arg(input: ParseStream) -> Result<OpArg> {
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

fn parse_indices(input: ParseStream) -> Result<Vec<IndexExpr>> {
    let content;
    syn::bracketed!(content in input);
    let mut indices = Vec::new();
    while !content.is_empty() {
        if content.peek(LitInt) {
            indices.push(IndexExpr::Lit(content.parse()?));
        } else {
            indices.push(IndexExpr::Ident(content.parse()?));
        }
        if content.peek(Token![,]) {
            content.parse::<Token![,]>()?;
        }
    }
    if indices.is_empty() {
        return Err(content.error("prefix access must include at least one index"));
    }
    Ok(indices)
}

fn var_ref_string(var_ref: &VarRef) -> String {
    let mut name = var_ref.name.to_string();
    if !var_ref.indices.is_empty() {
        let items = var_ref.indices.iter().map(|index| match index {
            IndexExpr::Ident(ident) => ident.to_string(),
            IndexExpr::Lit(lit) => lit.to_string(),
        });
        name.push('[');
        name.push_str(&items.collect::<Vec<_>>().join(","));
        name.push(']');
    }
    name
}

fn parse_op_attr_value(input: ParseStream) -> Result<OpAttrValue> {
    let negative = if input.peek(Token![-]) {
        input.parse::<Token![-]>()?;
        true
    } else {
        false
    };

    if input.peek(syn::LitFloat) {
        let lit: syn::LitFloat = input.parse()?;
        let mut value: f32 = lit.base10_parse()?;
        if negative {
            value = -value;
        }
        return Ok(OpAttrValue::Literal(value));
    }
    if input.peek(LitInt) {
        let lit: LitInt = input.parse()?;
        let mut value: f32 = lit.base10_parse::<i64>()? as f32;
        if negative {
            value = -value;
        }
        return Ok(OpAttrValue::Literal(value));
    }
    if input.peek(Ident) {
        let ident: Ident = input.parse()?;
        if negative {
            return Err(input.error("unexpected '-' before identifier"));
        }
        let name = ident.to_string();
        if name == "inf" {
            return Ok(OpAttrValue::Literal(f32::INFINITY));
        }
        return Ok(OpAttrValue::Var(ident));
    }

    Err(input.error("expected literal or identifier for op setting"))
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
                        MemoryKindToken::Constant => quote! { ::openinfer::MemoryKind::Constant },
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
                        let table_indices = var.table_indices.iter().map(|index| {
                            let s = index.to_string();
                            quote! { #s.to_string() }
                        });
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
                            );
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
                                let s = var_ref_string(i);
                                    quote! { #s.to_string() }
                                });
                                let output = op.output.to_string();
                                let attrs =
                                    validation::ops::op_attrs_expr(&op.name, &op.settings)?;
                                quote! {
                                    ::openinfer::NodeKind::Op {
                                        op: #kind,
                                        attrs: #attrs,
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
        "relu" => Ok(quote! { ::openinfer::OpKind::Relu }),
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
