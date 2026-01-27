use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::Path;

use syn::{
    Expr, ExprArray, ExprCall, ExprLit, ExprMatch, ExprPath, ExprReference, File, Item, ItemConst,
    ItemImpl, Lit, Pat, Type,
};

#[derive(Debug)]
struct OpSchemaInfo {
    kind_ident: String,
    inplace: bool,
    accumulate: bool,
    dtype_support: Option<String>,
}

pub fn generate_cpu_kernels(manifest_dir: &Path) -> Result<(), Box<dyn Error>> {
    let op_defs_path = manifest_dir.join("src/registry/op_defs.rs");
    let op_dtypes_path = manifest_dir.join("src/registry/op_dtypes.rs");
    let types_path = manifest_dir.join("src/graph/types.rs");

    let op_schemas = parse_ops(&op_defs_path)?;
    let opkind_map = parse_opkind_map(&types_path)?;
    let add_normal_dtypes = parse_dtype_list(&op_dtypes_path, "ADD_NORMAL_DTYPES")?;
    let acc_pairs = parse_dtype_pairs(&op_dtypes_path, "ACC_INT_PAIRS")?;

    for schema in op_schemas {
        let op_name = opkind_map
            .get(&schema.kind_ident)
            .ok_or_else(|| format!("missing OpKind::{} in OpKind::as_str()", schema.kind_ident))?;
        let dtype_support = schema.dtype_support.as_deref().unwrap_or("");
        if dtype_support != "ADD_DTYPE_SUPPORT" {
            return Err(format!(
                "unsupported dtype_support {dtype_support:?} for OpKind::{}, update generator/op_schema.rs",
                schema.kind_ident
            )
            .into());
        }
        let op_dir = manifest_dir.join(format!("src/ops/cpu/{op_name}"));
        write_kernel_rs(
            &op_dir,
            op_name,
            &add_normal_dtypes,
            &acc_pairs,
            schema.inplace,
            schema.accumulate,
        )?;
    }

    Ok(())
}

fn parse_ops(path: &Path) -> Result<Vec<OpSchemaInfo>, Box<dyn Error>> {
    let file = parse_file(path)?;
    for item in file.items {
        if let Item::Const(item_const) = item {
            if item_const.ident == "OPS" {
                return parse_ops_const(&item_const);
            }
        }
    }
    Err(format!("missing OPS const in {}", path.display()).into())
}

fn parse_ops_const(item_const: &ItemConst) -> Result<Vec<OpSchemaInfo>, Box<dyn Error>> {
    let expr = unwrap_reference(item_const.expr.as_ref())?;
    let array = match expr {
        Expr::Array(array) => array,
        _ => return Err("OPS const is not an array literal".into()),
    };

    let mut ops = Vec::new();
    for elem in &array.elems {
        let expr = unwrap_reference(elem)?;
        if let Expr::Struct(struct_expr) = expr {
            let mut kind_ident = None;
            let mut inplace = false;
            let mut accumulate = false;
            let mut dtype_support = None;
            for field in &struct_expr.fields {
                let field_name = match &field.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(_) => continue,
                };
                match field_name.as_str() {
                    "kind" => {
                        kind_ident = extract_path_ident(&field.expr);
                    }
                    "inplace" => {
                        inplace = extract_allow_flag(&field.expr)?;
                    }
                    "accumulate" => {
                        accumulate = extract_allow_flag(&field.expr)?;
                    }
                    "dtype_support" => {
                        dtype_support = extract_dtype_support(&field.expr)?;
                    }
                    _ => {}
                }
            }
            let kind_ident = kind_ident.ok_or("OpSchema missing kind")?;
            ops.push(OpSchemaInfo {
                kind_ident,
                inplace,
                accumulate,
                dtype_support,
            });
        }
    }
    Ok(ops)
}

fn parse_opkind_map(path: &Path) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let file = parse_file(path)?;
    for item in file.items {
        if let Item::Impl(ItemImpl { self_ty, items, .. }) = item {
            if is_opkind_impl(&self_ty) {
                for impl_item in items {
                    if let syn::ImplItem::Fn(func) = impl_item {
                        if func.sig.ident == "as_str" {
                            return extract_match_map(&func.block);
                        }
                    }
                }
            }
        }
    }
    Err("missing OpKind::as_str implementation".into())
}

fn parse_dtype_list(path: &Path, name: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let file = parse_file(path)?;
    for item in file.items {
        if let Item::Const(item_const) = item {
            if item_const.ident == name {
                let expr = unwrap_reference(item_const.expr.as_ref())?;
                let array = match expr {
                    Expr::Array(array) => array,
                    _ => return Err(format!("{name} const is not an array literal").into()),
                };
                return extract_dtype_array(&array);
            }
        }
    }
    Err(format!("missing {name} const in {}", path.display()).into())
}

fn parse_dtype_pairs(path: &Path, name: &str) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let file = parse_file(path)?;
    for item in file.items {
        if let Item::Const(item_const) = item {
            if item_const.ident == name {
                let expr = unwrap_reference(item_const.expr.as_ref())?;
                let array = match expr {
                    Expr::Array(array) => array,
                    _ => return Err(format!("{name} const is not an array literal").into()),
                };
                return extract_dtype_pairs(&array);
            }
        }
    }
    Err(format!("missing {name} const in {}", path.display()).into())
}

fn parse_file(path: &Path) -> Result<File, Box<dyn Error>> {
    let contents = fs::read_to_string(path)?;
    syn::parse_file(&contents).map_err(|err| format!("failed to parse {}: {err}", path.display()).into())
}

fn unwrap_reference(expr: &Expr) -> Result<&Expr, Box<dyn Error>> {
    match expr {
        Expr::Reference(ExprReference { expr, .. }) => Ok(expr.as_ref()),
        other => Ok(other),
    }
}

fn extract_path_ident(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(ExprPath { path, .. }) => path.segments.last().map(|seg| seg.ident.to_string()),
        _ => None,
    }
}

fn extract_allow_flag(expr: &Expr) -> Result<bool, Box<dyn Error>> {
    let ident = extract_path_ident(expr).ok_or("expected path for support flag")?;
    match ident.as_str() {
        "Allow" => Ok(true),
        "Deny" => Ok(false),
        _ => Err(format!("unexpected support flag {ident}").into()),
    }
}

fn extract_dtype_support(expr: &Expr) -> Result<Option<String>, Box<dyn Error>> {
    match expr {
        Expr::Path(ExprPath { path, .. }) => {
            let last = path.segments.last().map(|seg| seg.ident.to_string());
            if last.as_deref() == Some("None") {
                Ok(None)
            } else {
                Err("unexpected dtype_support path".into())
            }
        }
        Expr::Call(ExprCall { func, args, .. }) => {
            let func_ident = extract_path_ident(func).ok_or("dtype_support call missing ident")?;
            if func_ident != "Some" {
                return Err("dtype_support call is not Some".into());
            }
            let arg = args.first().ok_or("dtype_support Some missing arg")?;
            let arg_expr = match arg {
                Expr::Reference(ExprReference { expr, .. }) => expr.as_ref(),
                other => other,
            };
            extract_path_ident(arg_expr).map(Some).ok_or_else(|| "invalid dtype_support arg".into())
        }
        _ => Err("unsupported dtype_support expr".into()),
    }
}

fn is_opkind_impl(self_ty: &Box<Type>) -> bool {
    match self_ty.as_ref() {
        Type::Path(path) => path
            .path
            .segments
            .last()
            .map(|seg| seg.ident == "OpKind")
            .unwrap_or(false),
        _ => false,
    }
}

fn extract_match_map(block: &syn::Block) -> Result<HashMap<String, String>, Box<dyn Error>> {
    for stmt in &block.stmts {
        let expr = match stmt {
            syn::Stmt::Expr(expr, _) => expr,
            _ => continue,
        };
        if let Expr::Match(match_expr) = expr {
            return extract_match_arms(match_expr);
        }
    }
    Err("OpKind::as_str missing match".into())
}

fn extract_match_arms(match_expr: &ExprMatch) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let mut map = HashMap::new();
    for arm in &match_expr.arms {
        let ident = match &arm.pat {
            Pat::Path(path) => path.path.segments.last().map(|seg| seg.ident.to_string()),
            _ => None,
        }
        .ok_or("unexpected match arm pattern in OpKind::as_str")?;
        let value = match &*arm.body {
            Expr::Lit(ExprLit {
                lit: Lit::Str(lit),
                ..
            }) => lit.value(),
            _ => return Err("unexpected match arm body in OpKind::as_str".into()),
        };
        map.insert(ident, value);
    }
    Ok(map)
}

fn extract_dtype_array(array: &ExprArray) -> Result<Vec<String>, Box<dyn Error>> {
    let mut dtypes = Vec::new();
    for elem in &array.elems {
        let dtype = extract_path_ident(elem).ok_or("unexpected dtype entry")?;
        dtypes.push(dtype);
    }
    Ok(dtypes)
}

fn extract_dtype_pairs(array: &ExprArray) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let mut pairs = Vec::new();
    for elem in &array.elems {
        let tuple = match elem {
            Expr::Tuple(tuple) => tuple,
            _ => return Err("unexpected dtype pair entry".into()),
        };
        if tuple.elems.len() != 2 {
            return Err("dtype pair must have two elements".into());
        }
        let first = extract_path_ident(&tuple.elems[0]).ok_or("missing dtype pair lhs")?;
        let second = extract_path_ident(&tuple.elems[1]).ok_or("missing dtype pair rhs")?;
        pairs.push((first, second));
    }
    Ok(pairs)
}

fn write_kernel_rs(
    op_dir: &Path,
    op_name: &str,
    normal_dtypes: &[String],
    acc_pairs: &[(String, String)],
    inplace: bool,
    accumulate: bool,
) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(op_dir)?;
    let mut out = String::new();
    out.push_str("// @generated by build.rs. Do not edit.\n");
    out.push_str("use anyhow::{anyhow, Result};\n\n");
    out.push_str("use crate::graph::OpAttrs;\n");
    out.push_str("use crate::ops::cpu::registry::expect_output;\n");
    out.push_str("use crate::tensor::TensorValue;\n\n");

    out.push_str(&format!(
        "pub fn {op_name}_normal_dispatch(_attrs: &OpAttrs, inputs: &[TensorValue], output: Option<&mut TensorValue>) -> Result<()> {{\n"
    ));
    out.push_str("    let out = expect_output(output)?;\n");
    out.push_str("    match (&inputs[0], &inputs[1], out) {\n");
    for dtype in normal_dtypes {
        let variant = dtype;
        let suffix = dtype_suffix(dtype)?;
        if is_packed(dtype) {
            out.push_str(&format!(
                "        (TensorValue::{variant}(a), TensorValue::{variant}(b), TensorValue::{variant}(out)) => super::kernels::packed::{op_name}_{suffix}_packed(a, b, out),\n"
            ));
        } else {
            out.push_str(&format!(
                "        (TensorValue::{variant}(a), TensorValue::{variant}(b), TensorValue::{variant}(out)) => super::kernels::normal::{op_name}_{suffix}_normal(a, b, out),\n"
            ));
        }
    }
    out.push_str("        _ => Err(anyhow!(\"dtype mismatch\")),\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    if inplace {
        out.push_str(&format!(
            "pub fn {op_name}_inplace_dispatch(_attrs: &OpAttrs, inputs: &[TensorValue], output: Option<&mut TensorValue>) -> Result<()> {{\n"
        ));
        out.push_str("    let out = expect_output(output)?;\n");
        out.push_str("    match (out, &inputs[1]) {\n");
        for dtype in normal_dtypes {
            let variant = dtype;
            let suffix = dtype_suffix(dtype)?;
            if is_packed(dtype) {
                out.push_str(&format!(
                    "        (TensorValue::{variant}(a), TensorValue::{variant}(b)) => super::kernels::packed::{op_name}_{suffix}_packed_inplace(a, b),\n"
                ));
            } else {
                out.push_str(&format!(
                    "        (TensorValue::{variant}(a), TensorValue::{variant}(b)) => super::kernels::normal::{op_name}_{suffix}_inplace(a, b),\n"
                ));
            }
        }
        out.push_str("        _ => Err(anyhow!(\"dtype mismatch\")),\n");
        out.push_str("    }\n");
        out.push_str("}\n\n");
    }

    if accumulate {
        out.push_str(&format!(
            "pub fn {op_name}_accumulate_dispatch(_attrs: &OpAttrs, inputs: &[TensorValue], output: Option<&mut TensorValue>) -> Result<()> {{\n"
        ));
        out.push_str("    let out = expect_output(output)?;\n");
        out.push_str("    match (&inputs[0], &inputs[1], out) {\n");
        for (input, acc) in acc_pairs {
            let input_variant = input;
            let acc_variant = acc;
            let input_suffix = dtype_suffix(input)?;
            let acc_suffix = dtype_suffix(acc)?;
            out.push_str(&format!(
                "        (TensorValue::{input_variant}(a), TensorValue::{input_variant}(b), TensorValue::{acc_variant}(out)) => super::kernels::accumulate::{op_name}_{input_suffix}_accumulate_{acc_suffix}(a, b, out),\n"
            ));
        }
        out.push_str("        _ => Err(anyhow!(\"dtype mismatch\")),\n");
        out.push_str("    }\n");
        out.push_str("}\n");
    }

    let out_path = op_dir.join("kernel.rs");
    fs::write(out_path, out)?;
    Ok(())
}

fn is_packed(dtype: &str) -> bool {
    matches!(dtype, "I1" | "I2" | "I4" | "U1" | "U2" | "U4")
}

fn dtype_suffix(dtype: &str) -> Result<&'static str, Box<dyn Error>> {
    let suffix = match dtype {
        "F8" => "f8",
        "BF16" => "bf16",
        "F16" => "f16",
        "F32" => "f32",
        "F64" => "f64",
        "I1" => "i1",
        "I2" => "i2",
        "I4" => "i4",
        "I8" => "i8",
        "I16" => "i16",
        "I32" => "i32",
        "I64" => "i64",
        "U1" => "u1",
        "U2" => "u2",
        "U4" => "u4",
        "U8" => "u8",
        "U16" => "u16",
        "U32" => "u32",
        "U64" => "u64",
        "Bool" => "bool",
        "Bitset" => "bitset",
        _ => return Err(format!("unsupported dtype {dtype}").into()),
    };
    Ok(suffix)
}
