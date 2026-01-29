use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use syn::{
    Expr, ExprArray, ExprCall, ExprLit, ExprMatch, ExprPath, ExprReference, File, Item, ItemConst,
    ItemImpl, Lit, Pat, Type,
};

#[derive(Debug)]
struct OpSchemaInfo {
    kind_ident: String,
    inputs: InputArityInfo,
    inplace: bool,
    accumulate: bool,
    dtype_support: Option<String>,
    uses_attrs: bool,
    fixed_output: Option<String>,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum InputArityInfo {
    Fixed(usize),
    AtLeast(usize),
    Any,
}

impl InputArityInfo {
    fn fixed(self) -> Option<usize> {
        match self {
            InputArityInfo::Fixed(count) => Some(count),
            _ => None,
        }
    }
}

pub fn generate_cpu_kernels(manifest_dir: &Path) -> Result<(), Box<dyn Error>> {
    let op_defs_path = manifest_dir.join("src/registry/op_defs.rs");
    let types_path = manifest_dir.join("src/graph/types.rs");

    let op_schemas = parse_ops(&op_defs_path)?;
    let opkind_map = parse_opkind_map(&types_path)?;
    let op_dtype_paths = collect_op_dtype_paths(manifest_dir)?;

    for schema in op_schemas {
        let op_name = opkind_map
            .get(&schema.kind_ident)
            .ok_or_else(|| format!("missing OpKind::{} in OpKind::as_str()", schema.kind_ident))?;
        let dtype_support = schema
            .dtype_support
            .as_deref()
            .ok_or_else(|| format!("missing dtype_support for OpKind::{}", schema.kind_ident))?;
        let inputs = schema
            .inputs
            .fixed()
            .ok_or_else(|| format!("non-fixed input arity for OpKind::{}", schema.kind_ident))?;
        let (normal_name, acc_name) = find_dtype_support_fields(&op_dtype_paths, dtype_support)?;
        let normal_dtypes = parse_dtype_list_in_files(&op_dtype_paths, &normal_name)?;
        let acc_pairs = parse_dtype_pairs_in_files(&op_dtype_paths, &acc_name)?;
        let op_dir = manifest_dir.join(format!("src/ops/cpu/{op_name}"));
        write_kernel_rs(
            &op_dir,
            op_name,
            &normal_dtypes,
            &acc_pairs,
            inputs,
            schema.inplace,
            schema.accumulate,
            schema.uses_attrs,
            schema.fixed_output.as_deref(),
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
            let mut inputs = None;
            let mut inplace = false;
            let mut accumulate = false;
            let mut dtype_support = None;
            let mut uses_attrs = false;
            let mut fixed_output = None;
            for field in &struct_expr.fields {
                let field_name = match &field.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(_) => continue,
                };
                match field_name.as_str() {
                    "kind" => {
                        kind_ident = extract_path_ident(&field.expr);
                    }
                    "inputs" => {
                        inputs = Some(extract_input_arity(&field.expr)?);
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
                    "type_rule" => {
                        fixed_output = extract_fixed_output(&field.expr)?;
                    }
                    "attrs" => {
                        uses_attrs = extract_attrs_use(&field.expr)?;
                    }
                    _ => {}
                }
            }
            let kind_ident = kind_ident.ok_or("OpSchema missing kind")?;
            let inputs = inputs.ok_or("OpSchema missing inputs")?;
            ops.push(OpSchemaInfo {
                kind_ident,
                inputs,
                inplace,
                accumulate,
                dtype_support,
                uses_attrs,
                fixed_output,
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

fn parse_dtype_list_in_files(paths: &[PathBuf], name: &str) -> Result<Vec<String>, Box<dyn Error>> {
    for path in paths {
        if let Ok(list) = parse_dtype_list(path, name) {
            return Ok(list);
        }
    }
    Err(format!("missing {name} const in registry/op_dtypes").into())
}

fn parse_dtype_pairs_in_files(
    paths: &[PathBuf],
    name: &str,
) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    for path in paths {
        if let Ok(pairs) = parse_dtype_pairs(path, name) {
            return Ok(pairs);
        }
    }
    Err(format!("missing {name} const in registry/op_dtypes").into())
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

fn extract_input_arity(expr: &Expr) -> Result<InputArityInfo, Box<dyn Error>> {
    match expr {
        Expr::Path(ExprPath { path, .. }) => {
            let last = path.segments.last().map(|seg| seg.ident.to_string());
            match last.as_deref() {
                Some("Any") => Ok(InputArityInfo::Any),
                _ => Err("unexpected InputArity path".into()),
            }
        }
        Expr::Call(ExprCall { func, args, .. }) => {
            let func_ident = extract_path_ident(func).ok_or("InputArity call missing ident")?;
            match func_ident.as_str() {
                "Fixed" => Ok(InputArityInfo::Fixed(extract_usize_arg(args.first())?)),
                "AtLeast" => Ok(InputArityInfo::AtLeast(extract_usize_arg(args.first())?)),
                _ => Err(format!("unexpected InputArity call {func_ident}").into()),
            }
        }
        _ => Err("unsupported InputArity expr".into()),
    }
}

fn extract_usize_arg(arg: Option<&Expr>) -> Result<usize, Box<dyn Error>> {
    let expr = arg.ok_or("missing usize arg")?;
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(lit),
            ..
        }) => lit.base10_parse::<usize>().map_err(|_| "invalid usize literal".into()),
        _ => Err("expected usize literal".into()),
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

fn extract_attrs_use(expr: &Expr) -> Result<bool, Box<dyn Error>> {
    let expr = unwrap_reference(expr)?;
    match expr {
        Expr::Array(ExprArray { elems, .. }) => {
            if elems.is_empty() {
                return Ok(false);
            }
            for elem in elems {
                let elem = unwrap_reference(elem)?;
                let ident = extract_path_ident(elem).ok_or("invalid attrs element")?;
                if ident != "ACC_ATTR" {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        _ => Err("unsupported attrs expr".into()),
    }
}

fn extract_fixed_output(expr: &Expr) -> Result<Option<String>, Box<dyn Error>> {
    let expr = unwrap_reference(expr)?;
    match expr {
        Expr::Call(ExprCall { func, args, .. }) => {
            let func_ident = extract_path_ident(func).ok_or("type_rule call missing ident")?;
            if func_ident != "Fixed" {
                return Ok(None);
            }
            let arg = args.first().ok_or("type_rule Fixed missing arg")?;
            let arg_expr = match arg {
                Expr::Reference(ExprReference { expr, .. }) => expr.as_ref(),
                other => other,
            };
            let dtype_ident = extract_path_ident(arg_expr).ok_or("invalid Fixed dtype")?;
            Ok(Some(dtype_ident))
        }
        Expr::Path(_) => Ok(None),
        _ => Ok(None),
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

fn collect_op_dtype_paths(manifest_dir: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut paths = Vec::new();
    let root = manifest_dir.join("src/registry/op_dtypes.rs");
    paths.push(root);
    let dir = manifest_dir.join("src/registry/op_dtypes");
    if dir.exists() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
                paths.push(path);
            }
        }
    }
    Ok(paths)
}

fn find_dtype_support_fields(
    paths: &[PathBuf],
    support_name: &str,
) -> Result<(String, String), Box<dyn Error>> {
    for path in paths {
        let file = parse_file(path)?;
        for item in file.items {
            let item_const = match item {
                Item::Const(item_const) => item_const,
                _ => continue,
            };
            if item_const.ident != support_name {
                continue;
            }
            let expr = unwrap_reference(item_const.expr.as_ref())?;
            let struct_expr = match expr {
                Expr::Struct(struct_expr) => struct_expr,
                _ => return Err(format!("{support_name} const is not a struct literal").into()),
            };
            let mut normal = None;
            let mut accumulate = None;
            for field in &struct_expr.fields {
                let field_name = match &field.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(_) => continue,
                };
                match field_name.as_str() {
                    "normal" => {
                        let expr = unwrap_reference(&field.expr)?;
                        normal = extract_path_ident(expr);
                    }
                    "accumulate" => {
                        let expr = unwrap_reference(&field.expr)?;
                        accumulate = extract_path_ident(expr);
                    }
                    _ => {}
                }
            }
            let normal = normal.ok_or("OpDTypeSupport missing normal field")?;
            let accumulate = accumulate.ok_or("OpDTypeSupport missing accumulate field")?;
            return Ok((normal, accumulate));
        }
    }
    Err(format!("missing {support_name} const in registry/op_dtypes").into())
}

fn write_kernel_rs(
    op_dir: &Path,
    op_name: &str,
    normal_dtypes: &[String],
    acc_pairs: &[(String, String)],
    inputs: usize,
    inplace: bool,
    accumulate: bool,
    uses_attrs: bool,
    fixed_output: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    if inputs != 1 && inputs != 2 {
        return Err(format!("unsupported input count {inputs} for {op_name}").into());
    }
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
    if inputs == 1 {
        out.push_str("    match (&inputs[0], out) {\n");
        for dtype in normal_dtypes {
            let variant = dtype;
            let out_variant = fixed_output.unwrap_or(variant);
            let suffix = dtype_suffix(dtype)?;
            if is_packed(dtype) {
                if uses_attrs {
                    out.push_str(&format!(
                        "        (TensorValue::{variant}(a), TensorValue::{out_variant}(out)) => super::kernels::packed::{op_name}_{suffix}_packed(_attrs, a, out),\n"
                    ));
                } else {
                    out.push_str(&format!(
                        "        (TensorValue::{variant}(a), TensorValue::{out_variant}(out)) => super::kernels::packed::{op_name}_{suffix}_packed(a, out),\n"
                    ));
                }
            } else {
                if uses_attrs {
                    out.push_str(&format!(
                        "        (TensorValue::{variant}(a), TensorValue::{out_variant}(out)) => super::kernels::normal::{op_name}_{suffix}_normal(_attrs, a, out),\n"
                    ));
                } else {
                    out.push_str(&format!(
                        "        (TensorValue::{variant}(a), TensorValue::{out_variant}(out)) => super::kernels::normal::{op_name}_{suffix}_normal(a, out),\n"
                    ));
                }
            }
        }
    } else {
        out.push_str("    match (&inputs[0], &inputs[1], out) {\n");
        for dtype in normal_dtypes {
            let variant = dtype;
            let out_variant = fixed_output.unwrap_or(variant);
            let suffix = dtype_suffix(dtype)?;
            if is_packed(dtype) {
                if uses_attrs {
                    out.push_str(&format!(
                        "        (TensorValue::{variant}(a), TensorValue::{variant}(b), TensorValue::{out_variant}(out)) => super::kernels::packed::{op_name}_{suffix}_packed(_attrs, a, b, out),\n"
                    ));
                } else {
                    out.push_str(&format!(
                        "        (TensorValue::{variant}(a), TensorValue::{variant}(b), TensorValue::{out_variant}(out)) => super::kernels::packed::{op_name}_{suffix}_packed(a, b, out),\n"
                    ));
                }
            } else {
                if uses_attrs {
                    out.push_str(&format!(
                        "        (TensorValue::{variant}(a), TensorValue::{variant}(b), TensorValue::{out_variant}(out)) => super::kernels::normal::{op_name}_{suffix}_normal(_attrs, a, b, out),\n"
                    ));
                } else {
                    out.push_str(&format!(
                        "        (TensorValue::{variant}(a), TensorValue::{variant}(b), TensorValue::{out_variant}(out)) => super::kernels::normal::{op_name}_{suffix}_normal(a, b, out),\n"
                    ));
                }
            }
        }
    }
    out.push_str("        _ => Err(anyhow!(\"dtype mismatch\")),\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    if inplace {
        let inputs_param = if inputs == 1 { "_inputs" } else { "inputs" };
        out.push_str(&format!(
            "pub fn {op_name}_inplace_dispatch(_attrs: &OpAttrs, {inputs_param}: &[TensorValue], output: Option<&mut TensorValue>) -> Result<()> {{\n"
        ));
        out.push_str("    let out = expect_output(output)?;\n");
        if inputs == 1 {
            out.push_str("    match out {\n");
            for dtype in normal_dtypes {
                let variant = dtype;
                let suffix = dtype_suffix(dtype)?;
                if is_packed(dtype) {
                    if uses_attrs {
                        out.push_str(&format!(
                            "        TensorValue::{variant}(a) => super::kernels::packed::{op_name}_{suffix}_packed_inplace(_attrs, a),\n"
                        ));
                    } else {
                        out.push_str(&format!(
                            "        TensorValue::{variant}(a) => super::kernels::packed::{op_name}_{suffix}_packed_inplace(a),\n"
                        ));
                    }
                } else {
                    if uses_attrs {
                        out.push_str(&format!(
                            "        TensorValue::{variant}(a) => super::kernels::normal::{op_name}_{suffix}_inplace(_attrs, a),\n"
                        ));
                    } else {
                        out.push_str(&format!(
                            "        TensorValue::{variant}(a) => super::kernels::normal::{op_name}_{suffix}_inplace(a),\n"
                        ));
                    }
                }
            }
        } else {
            out.push_str("    match (out, &inputs[1]) {\n");
            for dtype in normal_dtypes {
                let variant = dtype;
                let suffix = dtype_suffix(dtype)?;
                if is_packed(dtype) {
                    if uses_attrs {
                        out.push_str(&format!(
                            "        (TensorValue::{variant}(a), TensorValue::{variant}(b)) => super::kernels::packed::{op_name}_{suffix}_packed_inplace(_attrs, a, b),\n"
                        ));
                    } else {
                        out.push_str(&format!(
                            "        (TensorValue::{variant}(a), TensorValue::{variant}(b)) => super::kernels::packed::{op_name}_{suffix}_packed_inplace(a, b),\n"
                        ));
                    }
                } else {
                    if uses_attrs {
                        out.push_str(&format!(
                            "        (TensorValue::{variant}(a), TensorValue::{variant}(b)) => super::kernels::normal::{op_name}_{suffix}_inplace(_attrs, a, b),\n"
                        ));
                    } else {
                        out.push_str(&format!(
                            "        (TensorValue::{variant}(a), TensorValue::{variant}(b)) => super::kernels::normal::{op_name}_{suffix}_inplace(a, b),\n"
                        ));
                    }
                }
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
        if inputs == 1 {
            out.push_str("    match (&inputs[0], out) {\n");
            for (input, acc) in acc_pairs {
                let input_variant = input;
                let acc_variant = acc;
                let input_suffix = dtype_suffix(input)?;
                let acc_suffix = dtype_suffix(acc)?;
                out.push_str(&format!(
                    "        (TensorValue::{input_variant}(a), TensorValue::{acc_variant}(out)) => super::kernels::accumulate::{op_name}_{input_suffix}_accumulate_{acc_suffix}(a, out),\n"
                ));
            }
        } else {
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
