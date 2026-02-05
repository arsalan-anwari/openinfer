//! Generate CPU kernel sources from `ops.json`.
#![allow(dead_code)]
#![allow(unused_imports)]

use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;
use syn::{
    Expr, ExprArray, ExprCall, ExprLit, ExprMatch, ExprPath, ExprReference, File, Item, ItemConst,
    ItemImpl, Lit, Pat, Type,
};

#[derive(Debug)]
struct OpSchemaInfo {
    name: String,
    category: String,
    inputs: InputArityInfo,
    inplace: bool,
    accumulate: bool,
    dtype_support_ref: String,
    uses_attrs: bool,
    fixed_output: Option<String>,
    output_from_attr: bool,
    output_dtypes_ref: Option<String>,
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

/// Generate CPU kernel Rust sources based on `ops.json`.
pub fn generate_cpu_kernels(manifest_dir: &Path) -> Result<(), Box<dyn Error>> {
    let ops_json = load_ops_json(manifest_dir)?;

    for schema in ops_json.ops {
        let dtype_support = ops_json
            .dtype_sets
            .get(&schema.dtype_support_ref)
            .ok_or_else(|| format!("missing dtype support {}", schema.dtype_support_ref))?;
        let inputs = schema
            .inputs
            .fixed()
            .ok_or_else(|| format!("non-fixed input arity for op {}", schema.name))?;
        let normal_dtypes = &dtype_support.normal;
        let acc_pairs = &dtype_support.accumulate;
        let op_dir = cpu_op_dir(manifest_dir, &schema.category, &schema.name)?;
        if schema.output_from_attr {
            let output_ref = schema.output_dtypes_ref.as_deref().ok_or_else(|| {
                format!("missing output_dtypes_ref for op {}", schema.name)
            })?;
            let output_dtypes = ops_json
                .output_dtype_sets
                .get(output_ref)
                .ok_or_else(|| format!("missing output dtype set {output_ref}"))?;
            write_cast_kernel_rs(&op_dir, &schema.name, normal_dtypes, output_dtypes)?;
            continue;
        }
        write_kernel_rs(
            &op_dir,
            &schema.name,
            normal_dtypes,
            acc_pairs,
            inputs,
            schema.inplace,
            schema.accumulate,
            schema.uses_attrs,
            schema.fixed_output.as_deref(),
        )?;
    }

    Ok(())
}

#[derive(Debug)]
struct DTypeSupportSet {
    normal: Vec<String>,
    accumulate: Vec<(String, String)>,
}

#[derive(Debug)]
struct OpsJsonInfo {
    ops: Vec<OpSchemaInfo>,
    dtype_sets: HashMap<String, DTypeSupportSet>,
    output_dtype_sets: HashMap<String, Vec<String>>,
}

fn load_ops_json(manifest_dir: &Path) -> Result<OpsJsonInfo, Box<dyn Error>> {
    let ops_path = manifest_dir.join("../ops.json");
    let contents = fs::read_to_string(&ops_path)?;
    let root: Value = serde_json::from_str(&contents)?;

    let dtype_sets = parse_dtype_sets(root.get("dtype_sets"))?;
    let output_dtype_sets = parse_output_dtype_sets(root.get("output_dtype_sets"))?;
    let ops = parse_ops_from_json(root.get("ops"))?;

    Ok(OpsJsonInfo {
        ops,
        dtype_sets,
        output_dtype_sets,
    })
}

fn parse_ops_from_json(value: Option<&Value>) -> Result<Vec<OpSchemaInfo>, Box<dyn Error>> {
    let ops = value
        .and_then(|v| v.as_array())
        .ok_or_else(|| "ops.json missing ops array".to_string())?;
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        let obj = op
            .as_object()
            .ok_or_else(|| "ops.json op must be an object".to_string())?;
        let name = get_string(obj.get("name"), "op name")?;
        let category = get_string(obj.get("category"), "op category")?;
        let inputs = parse_input_arity_json(obj.get("inputs"))?;
        let inplace = parse_allow(obj.get("inplace"), "inplace")?;
        let accumulate = parse_allow(obj.get("accumulate"), "accumulate")?;
        let dtype_support_ref = get_string(obj.get("dtype_support_ref"), "dtype_support_ref")?;
        let uses_attrs = obj
            .get("attrs")
            .and_then(|v| v.as_array())
            .map(|attrs| attrs.iter().any(|attr| attr.as_str() != Some("acc")))
            .unwrap_or(false);
        let (fixed_output, output_from_attr) = parse_type_rule_json(obj.get("type_rule"))?;
        let output_dtypes_ref = obj
            .get("output_dtypes_ref")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        out.push(OpSchemaInfo {
            name,
            category,
            inputs,
            inplace,
            accumulate,
            dtype_support_ref,
            uses_attrs,
            fixed_output,
            output_from_attr,
            output_dtypes_ref,
        });
    }
    Ok(out)
}

fn parse_input_arity_json(value: Option<&Value>) -> Result<InputArityInfo, Box<dyn Error>> {
    let obj = value
        .and_then(|v| v.as_object())
        .ok_or_else(|| "inputs must be an object".to_string())?;
    let arity = get_string(obj.get("arity"), "inputs.arity")?;
    match arity.as_str() {
        "fixed" => {
            let count = obj
                .get("count")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| "inputs.count missing for fixed arity".to_string())?;
            Ok(InputArityInfo::Fixed(count as usize))
        }
        "at_least" => {
            let count = obj
                .get("count")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| "inputs.count missing for at_least arity".to_string())?;
            Ok(InputArityInfo::AtLeast(count as usize))
        }
        "any" => Ok(InputArityInfo::Any),
        other => Err(format!("unknown input arity {other}").into()),
    }
}

fn parse_type_rule_json(value: Option<&Value>) -> Result<(Option<String>, bool), Box<dyn Error>> {
    let obj = value
        .and_then(|v| v.as_object())
        .ok_or_else(|| "type_rule must be an object".to_string())?;
    let kind = get_string(obj.get("kind"), "type_rule.kind")?;
    match kind.as_str() {
        "fixed" => {
            let dtype = get_string(obj.get("dtype"), "type_rule.dtype")?;
            Ok((Some(to_dtype_variant(&dtype)?), false))
        }
        "acc_from_attr" => Ok((None, true)),
        "same_as_input" => Ok((None, false)),
        other => Err(format!("unknown type_rule kind {other}").into()),
    }
}

fn parse_allow(value: Option<&Value>, label: &str) -> Result<bool, Box<dyn Error>> {
    let value = get_string(value, label)?;
    match value.as_str() {
        "allow" => Ok(true),
        "deny" => Ok(false),
        other => Err(format!("unknown {label} value {other}").into()),
    }
}

fn parse_dtype_sets(value: Option<&Value>) -> Result<HashMap<String, DTypeSupportSet>, Box<dyn Error>> {
    let obj = value
        .and_then(|v| v.as_object())
        .ok_or_else(|| "ops.json missing dtype_sets object".to_string())?;
    let mut out = HashMap::new();
    for (name, entry) in obj {
        let entry_obj = entry
            .as_object()
            .ok_or_else(|| format!("dtype_sets.{name} must be an object"))?;
        let normal = entry_obj
            .get("normal")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("dtype_sets.{name}.normal missing"))?
            .iter()
            .map(|v| v.as_str().ok_or_else(|| "dtype normal must be string".to_string()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(to_dtype_variant)
            .collect::<Result<Vec<_>, _>>()?;
        let accumulate = entry_obj
            .get("accumulate")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|pair| {
                        let pair_obj = pair.as_object().ok_or("accumulate pair must be object")?;
                        let input = get_string(pair_obj.get("input"), "accumulate.input")?;
                        let acc = get_string(pair_obj.get("acc"), "accumulate.acc")?;
                        Ok((to_dtype_variant(&input)?, to_dtype_variant(&acc)?))
                    })
                    .collect::<Result<Vec<_>, Box<dyn Error>>>()
            })
            .unwrap_or_else(|| Ok(Vec::new()))?;
        out.insert(
            name.clone(),
            DTypeSupportSet {
                normal,
                accumulate,
            },
        );
    }
    Ok(out)
}

fn parse_output_dtype_sets(
    value: Option<&Value>,
) -> Result<HashMap<String, Vec<String>>, Box<dyn Error>> {
    let mut out = HashMap::new();
    let Some(obj) = value.and_then(|v| v.as_object()) else {
        return Ok(out);
    };
    for (name, entry) in obj {
        let dtypes = entry
            .as_array()
            .ok_or_else(|| format!("output_dtype_sets.{name} must be array"))?
            .iter()
            .map(|v| v.as_str().ok_or_else(|| "output dtype must be string".to_string()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(to_dtype_variant)
            .collect::<Result<Vec<_>, _>>()?;
        out.insert(name.clone(), dtypes);
    }
    Ok(out)
}

fn get_string(value: Option<&Value>, label: &str) -> Result<String, Box<dyn Error>> {
    value
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| format!("missing {label}").into())
}

fn to_dtype_variant(ident: &str) -> Result<String, Box<dyn Error>> {
    let variant = match ident {
        "f8" => "F8",
        "bf16" => "BF16",
        "f16" => "F16",
        "f32" => "F32",
        "f64" => "F64",
        "i1" => "I1",
        "i2" => "I2",
        "i4" => "I4",
        "i8" => "I8",
        "i16" => "I16",
        "i32" => "I32",
        "i64" => "I64",
        "u1" => "U1",
        "u2" => "U2",
        "u4" => "U4",
        "u8" => "U8",
        "u16" => "U16",
        "u32" => "U32",
        "u64" => "U64",
        "bool" => "Bool",
        "bitset" => "Bitset",
        other => return Err(format!("unsupported dtype {other}").into()),
    };
    Ok(variant.to_string())
}

fn cpu_op_dir(
    manifest_dir: &Path,
    category: &str,
    op_name: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    Ok(manifest_dir.join(format!("src/ops/cpu/{category}/{op_name}")))
}

fn write_cast_kernel_rs(
    op_dir: &Path,
    op_name: &str,
    normal_dtypes: &[String],
    output_dtypes: &[String],
) -> Result<(), Box<dyn Error>> {
    if normal_dtypes.is_empty() {
        return Err(format!("missing input dtypes for {op_name}").into());
    }
    if output_dtypes.is_empty() {
        return Err(format!("missing output dtypes for {op_name}").into());
    }
    fs::create_dir_all(op_dir)?;
    let mut out = String::new();
    out.push_str("// @generated by build.rs. Do not edit.\n");
    out.push_str("use anyhow::{anyhow, Result};\n\n");
    out.push_str("use crate::graph::{AttrValue, OpAttrs};\n");
    out.push_str("use crate::ops::cpu::registry::expect_output;\n");
    out.push_str("use crate::tensor::{DType, TensorValue};\n\n");
    out.push_str("use super::kernels::{normal, packed};\n\n");

    out.push_str(&format!(
        "pub fn {op_name}_normal_dispatch(attrs: &OpAttrs, inputs: &[TensorValue], output: Option<&mut TensorValue>) -> Result<()> {{\n"
    ));
    out.push_str("    let out = expect_output(output)?;\n");
    out.push_str("    let to_dtype = get_to_dtype(attrs)?;\n");
    out.push_str("    let rounding = get_rounding_mode(attrs)?;\n");
    out.push_str("    let saturate = get_saturate(attrs)?;\n");
    out.push_str("    let in_dtype = inputs[0].dtype();\n");
    out.push_str("    if !is_allowed_cast(in_dtype, to_dtype) {\n");
    out.push_str("        return Err(anyhow!(\"unsupported cast from {:?} to {:?}\", in_dtype, to_dtype));\n");
    out.push_str("    }\n\n");
    out.push_str("    match (&inputs[0], out) {\n");
    for in_dtype in normal_dtypes {
        let in_suffix = dtype_suffix(in_dtype)?;
        let in_width = dtype_bit_width(in_dtype)?;
        let in_is_packed_signed = is_packed_signed(in_dtype);
        let in_is_packed_unsigned = is_packed_unsigned(in_dtype);
        let in_is_signed = is_signed_int_str(in_dtype);
        let in_is_unsigned = is_unsigned_int_str(in_dtype);
        let in_is_float = is_float_str(in_dtype);
        for out_dtype in output_dtypes {
            if !allow_cast_by_rule(
                in_is_packed_signed,
                in_is_packed_unsigned,
                in_is_signed,
                in_is_unsigned,
                in_is_float,
                in_width,
                out_dtype,
            )? {
                continue;
            }
            let out_suffix = dtype_suffix(out_dtype)?;
            if in_is_packed_signed {
                let width = in_width;
                let conversion = packed_signed_conversion(out_dtype)?;
                out.push_str(&format!(
                    "        (TensorValue::{in_dtype}(a0), TensorValue::{out_dtype}(out)) => packed::cast_packed_signed(a0, out, {width}, |v| {conversion}),\n"
                ));
            } else if in_is_packed_unsigned {
                let width = in_width;
                let conversion = packed_unsigned_conversion(out_dtype)?;
                out.push_str(&format!(
                    "        (TensorValue::{in_dtype}(a0), TensorValue::{out_dtype}(out)) => packed::cast_packed_unsigned(a0, out, {width}, |v| {conversion}),\n"
                ));
            } else {
                let needs_rounding =
                    is_signed_int_str(out_dtype) || is_unsigned_int_str(out_dtype);
                if needs_rounding {
                    out.push_str(&format!(
                        "        (TensorValue::{in_dtype}(a0), TensorValue::{out_dtype}(out)) => normal::cast_to_{out_suffix}(a0, out, normal::{in_suffix}_to_f64, rounding, saturate),\n"
                    ));
                } else {
                    out.push_str(&format!(
                        "        (TensorValue::{in_dtype}(a0), TensorValue::{out_dtype}(out)) => normal::cast_to_{out_suffix}(a0, out, normal::{in_suffix}_to_f64),\n"
                    ));
                }
            }
        }
    }
    out.push_str("        _ => Err(anyhow!(\"dtype mismatch\")),\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    out.push_str("fn get_to_dtype(attrs: &OpAttrs) -> Result<DType> {\n");
    out.push_str("    attrs\n");
    out.push_str("        .items\n");
    out.push_str("        .iter()\n");
    out.push_str("        .find(|attr| attr.name == \"to\")\n");
    out.push_str("        .ok_or_else(|| anyhow!(\"missing to attribute\"))\n");
    out.push_str("        .and_then(|attr| match &attr.value {\n");
    out.push_str("            AttrValue::DType(dtype) => Ok(*dtype),\n");
    out.push_str("            _ => Err(anyhow!(\"to attribute must be a dtype\")),\n");
    out.push_str("        })\n");
    out.push_str("}\n\n");

    out.push_str("fn get_rounding_mode(attrs: &OpAttrs) -> Result<fn(f64) -> f64> {\n");
    out.push_str("    let value = match attrs.items.iter().find(|attr| attr.name == \"rounding_mode\") {\n");
    out.push_str("        Some(attr) => attr.value.clone(),\n");
    out.push_str("        None => return Ok(normal::round_trunc),\n");
    out.push_str("    };\n");
    out.push_str("    match value {\n");
    out.push_str("        AttrValue::Str(mode) => match mode.as_str() {\n");
    out.push_str("            \"trunc\" => Ok(normal::round_trunc),\n");
    out.push_str("            \"floor\" => Ok(normal::round_floor),\n");
    out.push_str("            \"ceil\" => Ok(normal::round_ceil),\n");
    out.push_str("            \"nearest\" => Ok(normal::round_nearest),\n");
    out.push_str("            _ => Err(anyhow!(\"unsupported rounding_mode {}\", mode)),\n");
    out.push_str("        },\n");
    out.push_str("        _ => Err(anyhow!(\"rounding_mode attribute must be a string\")),\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    out.push_str("fn get_saturate(attrs: &OpAttrs) -> Result<bool> {\n");
    out.push_str("    let value = match attrs.items.iter().find(|attr| attr.name == \"saturate\") {\n");
    out.push_str("        Some(attr) => attr.value.clone(),\n");
    out.push_str("        None => return Ok(true),\n");
    out.push_str("    };\n");
    out.push_str("    match value {\n");
    out.push_str("        AttrValue::Bool(val) => Ok(val),\n");
    out.push_str("        AttrValue::Int(val) => Ok(val != 0),\n");
    out.push_str("        AttrValue::UInt(val) => Ok(val != 0),\n");
    out.push_str("        _ => Err(anyhow!(\"saturate attribute must be bool/int/uint\")),\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    out.push_str("pub fn is_allowed_cast(in_dtype: DType, out_dtype: DType) -> bool {\n");
    out.push_str("    if is_float(out_dtype) {\n");
    out.push_str("        return is_packed_signed(in_dtype)\n");
    out.push_str("            || is_packed_unsigned(in_dtype)\n");
    out.push_str("            || is_signed_int(in_dtype)\n");
    out.push_str("            || is_unsigned_int(in_dtype)\n");
    out.push_str("            || is_float(in_dtype);\n");
    out.push_str("    }\n\n");
    out.push_str("    if is_signed_int(out_dtype) {\n");
    out.push_str("        if is_float(in_dtype) {\n");
    out.push_str("            return true;\n");
    out.push_str("        }\n");
    out.push_str("        if is_packed_signed(in_dtype) {\n");
    out.push_str("            return out_dtype.bit_width() > in_dtype.bit_width();\n");
    out.push_str("        }\n");
    out.push_str("        if is_signed_int(in_dtype) {\n");
    out.push_str("            return out_dtype.bit_width() > in_dtype.bit_width();\n");
    out.push_str("        }\n");
    out.push_str("        return false;\n");
    out.push_str("    }\n\n");
    out.push_str("    if is_unsigned_int(out_dtype) {\n");
    out.push_str("        if is_float(in_dtype) {\n");
    out.push_str("            return true;\n");
    out.push_str("        }\n");
    out.push_str("        if is_packed_unsigned(in_dtype) {\n");
    out.push_str("            return out_dtype.bit_width() > in_dtype.bit_width();\n");
    out.push_str("        }\n");
    out.push_str("        if is_unsigned_int(in_dtype) {\n");
    out.push_str("            return out_dtype.bit_width() > in_dtype.bit_width();\n");
    out.push_str("        }\n");
    out.push_str("        return false;\n");
    out.push_str("    }\n\n");
    out.push_str("    false\n");
    out.push_str("}\n\n");

    out.push_str("fn is_float(dtype: DType) -> bool {\n");
    out.push_str("    matches!(dtype, DType::F8 | DType::F16 | DType::BF16 | DType::F32 | DType::F64)\n");
    out.push_str("}\n\n");
    out.push_str("fn is_signed_int(dtype: DType) -> bool {\n");
    out.push_str("    matches!(dtype, DType::I8 | DType::I16 | DType::I32 | DType::I64)\n");
    out.push_str("}\n\n");
    out.push_str("fn is_unsigned_int(dtype: DType) -> bool {\n");
    out.push_str("    matches!(dtype, DType::U8 | DType::U16 | DType::U32 | DType::U64)\n");
    out.push_str("}\n\n");
    out.push_str("fn is_packed_signed(dtype: DType) -> bool {\n");
    out.push_str("    matches!(dtype, DType::I1 | DType::I2 | DType::I4)\n");
    out.push_str("}\n\n");
    out.push_str("fn is_packed_unsigned(dtype: DType) -> bool {\n");
    out.push_str("    matches!(dtype, DType::U1 | DType::U2 | DType::U4)\n");
    out.push_str("}\n");

    let out_path = op_dir.join("kernel.rs");
    fs::write(out_path, out)?;
    Ok(())
}

fn parse_ops(_: &Path) -> Result<Vec<OpSchemaInfo>, Box<dyn Error>> {
    Err("deprecated: parse ops from ops.json instead".into())
}

fn parse_ops_const(_: &ItemConst) -> Result<Vec<OpSchemaInfo>, Box<dyn Error>> {
    Err("deprecated: parse ops from ops.json instead".into())
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

fn extract_type_rule_info(expr: &Expr) -> Result<(Option<String>, bool), Box<dyn Error>> {
    let expr = unwrap_reference(expr)?;
    match expr {
        Expr::Call(ExprCall { func, args, .. }) => {
            let func_ident = extract_path_ident(func).ok_or("type_rule call missing ident")?;
            if func_ident == "Fixed" {
                let arg = args.first().ok_or("type_rule Fixed missing arg")?;
                let arg_expr = match arg {
                    Expr::Reference(ExprReference { expr, .. }) => expr.as_ref(),
                    other => other,
                };
                let dtype_ident = extract_path_ident(arg_expr).ok_or("invalid Fixed dtype")?;
                return Ok((Some(dtype_ident), false));
            }
            Ok((None, false))
        }
        Expr::Struct(struct_expr) => {
            let ident = struct_expr
                .path
                .segments
                .last()
                .map(|seg| seg.ident.to_string())
                .ok_or("type_rule struct missing ident")?;
            if ident == "AccFromAttr" {
                return Ok((None, true));
            }
            Ok((None, false))
        }
        Expr::Path(_) => Ok((None, false)),
        _ => Ok((None, false)),
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
    let mut list = Vec::new();
    for elem in &array.elems {
        let elem = unwrap_reference(elem)?;
        let ident = extract_path_ident(elem).ok_or("unexpected dtype list entry")?;
        list.push(ident);
    }
    Ok(list)
}

fn extract_dtype_pairs(array: &ExprArray) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let mut pairs = Vec::new();
    for elem in &array.elems {
        let elem = unwrap_reference(elem)?;
        let Expr::Tuple(tuple) = elem else {
            return Err("unexpected dtype pair entry".into());
        };
        if tuple.elems.len() != 2 {
            return Err("dtype pair must have two elements".into());
        }
        let left = extract_path_ident(&tuple.elems[0]).ok_or("invalid dtype pair left")?;
        let right = extract_path_ident(&tuple.elems[1]).ok_or("invalid dtype pair right")?;
        pairs.push((left, right));
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
    if inputs == 0 {
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
    let input_refs: Vec<String> = (0..inputs).map(|i| format!("&inputs[{i}]")).collect();
    let match_head = if inputs == 1 {
        "    match (&inputs[0], out) {\n".to_string()
    } else {
        format!("    match ({}, out) {{\n", input_refs.join(", "))
    };
    out.push_str(&match_head);
    let arg_names: Vec<String> = (0..inputs).map(|i| format!("a{i}")).collect();
    for dtype in normal_dtypes {
        let variant = dtype;
        let out_variant = fixed_output.unwrap_or(variant);
        let suffix = dtype_suffix(dtype)?;
        let pattern_inputs = arg_names
            .iter()
            .map(|name| format!("TensorValue::{variant}({name})"))
            .collect::<Vec<_>>()
            .join(", ");
        let call_args = arg_names.join(", ");
        if is_packed(dtype) {
            if uses_attrs {
                out.push_str(&format!(
                    "        ({pattern_inputs}, TensorValue::{out_variant}(out)) => super::kernels::packed::{op_name}_{suffix}_packed(_attrs, {call_args}, out),\n"
                ));
            } else {
                out.push_str(&format!(
                    "        ({pattern_inputs}, TensorValue::{out_variant}(out)) => super::kernels::packed::{op_name}_{suffix}_packed({call_args}, out),\n"
                ));
            }
        } else if uses_attrs {
            out.push_str(&format!(
                "        ({pattern_inputs}, TensorValue::{out_variant}(out)) => super::kernels::normal::{op_name}_{suffix}_normal(_attrs, {call_args}, out),\n"
            ));
        } else {
            out.push_str(&format!(
                "        ({pattern_inputs}, TensorValue::{out_variant}(out)) => super::kernels::normal::{op_name}_{suffix}_normal({call_args}, out),\n"
            ));
        }
    }
    out.push_str("        _ => Err(anyhow!(\"dtype mismatch\")),\n");
    out.push_str("    }\n");
    out.push_str("}\n\n");

    let generate_inplace = inplace && fixed_output.is_none();
    if generate_inplace {
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
                } else if uses_attrs {
                    out.push_str(&format!(
                        "        TensorValue::{variant}(a) => super::kernels::normal::{op_name}_{suffix}_inplace(_attrs, a),\n"
                    ));
                } else {
                    out.push_str(&format!(
                        "        TensorValue::{variant}(a) => super::kernels::normal::{op_name}_{suffix}_inplace(a),\n"
                    ));
                }
            }
        } else {
            let inplace_refs: Vec<String> =
                (1..inputs).map(|i| format!("&inputs[{i}]")).collect();
            out.push_str(&format!("    match (out, {}) {{\n", inplace_refs.join(", ")));
            let arg_names: Vec<String> = (0..inputs).map(|i| format!("a{i}")).collect();
            for dtype in normal_dtypes {
                let variant = dtype;
                let suffix = dtype_suffix(dtype)?;
                let pattern_inputs = arg_names
                    .iter()
                    .map(|name| format!("TensorValue::{variant}({name})"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let call_args = arg_names.join(", ");
                if is_packed(dtype) {
                    if uses_attrs {
                        out.push_str(&format!(
                            "        ({pattern_inputs}) => super::kernels::packed::{op_name}_{suffix}_packed_inplace(_attrs, {call_args}),\n"
                        ));
                    } else {
                        out.push_str(&format!(
                            "        ({pattern_inputs}) => super::kernels::packed::{op_name}_{suffix}_packed_inplace({call_args}),\n"
                        ));
                    }
                } else if uses_attrs {
                    out.push_str(&format!(
                        "        ({pattern_inputs}) => super::kernels::normal::{op_name}_{suffix}_inplace(_attrs, {call_args}),\n"
                    ));
                } else {
                    out.push_str(&format!(
                        "        ({pattern_inputs}) => super::kernels::normal::{op_name}_{suffix}_inplace({call_args}),\n"
                    ));
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
        let input_refs: Vec<String> = (0..inputs).map(|i| format!("&inputs[{i}]")).collect();
        let match_head = if inputs == 1 {
            "    match (&inputs[0], out) {\n".to_string()
        } else {
            format!("    match ({}, out) {{\n", input_refs.join(", "))
        };
        out.push_str(&match_head);
        let arg_names: Vec<String> = (0..inputs).map(|i| format!("a{i}")).collect();
        for (input, acc) in acc_pairs {
            let input_variant = input;
            let acc_variant = acc;
            let input_suffix = dtype_suffix(input)?;
            let acc_suffix = dtype_suffix(acc)?;
            let pattern_inputs = arg_names
                .iter()
                .map(|name| format!("TensorValue::{input_variant}({name})"))
                .collect::<Vec<_>>()
                .join(", ");
            let call_args = arg_names.join(", ");
            if uses_attrs {
                out.push_str(&format!(
                    "        ({pattern_inputs}, TensorValue::{acc_variant}(out)) => super::kernels::accumulate::{op_name}_{input_suffix}_accumulate_{acc_suffix}(_attrs, {call_args}, out),\n"
                ));
            } else {
                out.push_str(&format!(
                    "        ({pattern_inputs}, TensorValue::{acc_variant}(out)) => super::kernels::accumulate::{op_name}_{input_suffix}_accumulate_{acc_suffix}({call_args}, out),\n"
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

fn dtype_bit_width(dtype: &str) -> Result<u8, Box<dyn Error>> {
    let width = match dtype {
        "I1" | "U1" => 1,
        "I2" | "U2" => 2,
        "I4" | "U4" => 4,
        "I8" | "U8" | "F8" => 8,
        "I16" | "U16" | "F16" | "BF16" => 16,
        "I32" | "U32" | "F32" => 32,
        "I64" | "U64" | "F64" => 64,
        _ => return Err(format!("unsupported dtype {dtype}").into()),
    };
    Ok(width)
}

fn is_float_str(dtype: &str) -> bool {
    matches!(dtype, "F8" | "F16" | "BF16" | "F32" | "F64")
}

fn is_signed_int_str(dtype: &str) -> bool {
    matches!(dtype, "I8" | "I16" | "I32" | "I64")
}

fn is_unsigned_int_str(dtype: &str) -> bool {
    matches!(dtype, "U8" | "U16" | "U32" | "U64")
}

fn is_packed_signed(dtype: &str) -> bool {
    matches!(dtype, "I1" | "I2" | "I4")
}

fn is_packed_unsigned(dtype: &str) -> bool {
    matches!(dtype, "U1" | "U2" | "U4")
}

fn allow_cast_by_rule(
    in_is_packed_signed: bool,
    in_is_packed_unsigned: bool,
    in_is_signed: bool,
    in_is_unsigned: bool,
    in_is_float: bool,
    in_width: u8,
    out_dtype: &str,
) -> Result<bool, Box<dyn Error>> {
    let out_is_float = is_float_str(out_dtype);
    if out_is_float {
        return Ok(
            in_is_packed_signed
                || in_is_packed_unsigned
                || in_is_signed
                || in_is_unsigned
                || in_is_float,
        );
    }

    let out_is_signed = is_signed_int_str(out_dtype);
    if out_is_signed {
        if in_is_float {
            return Ok(true);
        }
        if in_is_packed_signed || in_is_signed {
            let out_width = dtype_bit_width(out_dtype)?;
            return Ok(out_width > in_width);
        }
        return Ok(false);
    }

    let out_is_unsigned = is_unsigned_int_str(out_dtype);
    if out_is_unsigned {
        if in_is_float {
            return Ok(true);
        }
        if in_is_packed_unsigned || in_is_unsigned {
            let out_width = dtype_bit_width(out_dtype)?;
            return Ok(out_width > in_width);
        }
        return Ok(false);
    }

    Ok(false)
}

fn packed_signed_conversion(out_dtype: &str) -> Result<String, Box<dyn Error>> {
    let expr = match out_dtype {
        "I8" => "v".to_string(),
        "I16" => "v as i16".to_string(),
        "I32" => "v as i32".to_string(),
        "I64" => "v as i64".to_string(),
        "F8" => "normal::f64_to_f8(v as f64)".to_string(),
        "F16" => "normal::f64_to_f16(v as f64)".to_string(),
        "BF16" => "normal::f64_to_bf16(v as f64)".to_string(),
        "F32" => "v as f32".to_string(),
        "F64" => "v as f64".to_string(),
        _ => return Err(format!("unsupported packed signed output {out_dtype}").into()),
    };
    Ok(expr)
}

fn packed_unsigned_conversion(out_dtype: &str) -> Result<String, Box<dyn Error>> {
    let expr = match out_dtype {
        "U8" => "v".to_string(),
        "U16" => "v as u16".to_string(),
        "U32" => "v as u32".to_string(),
        "U64" => "v as u64".to_string(),
        "F8" => "normal::f64_to_f8(v as f64)".to_string(),
        "F16" => "normal::f64_to_f16(v as f64)".to_string(),
        "BF16" => "normal::f64_to_bf16(v as f64)".to_string(),
        "F32" => "v as f32".to_string(),
        "F64" => "v as f64".to_string(),
        _ => return Err(format!("unsupported packed unsigned output {out_dtype}").into()),
    };
    Ok(expr)
}
