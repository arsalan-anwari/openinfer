use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::thread;
use std::time::Duration;

use syn::{Expr, ExprPath, ExprReference, File, Item};

struct OpSpecInfo {
    name: String,
    inplace: bool,
    broadcast: bool,
    accumulate: bool,
}

fn main() {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("missing workspace root");
    let openinfer_dir = workspace_root.join("openinfer");
    let op_defs_path = openinfer_dir.join("src/registry/op_defs.rs");
    let specs = match parse_opspecs(&op_defs_path) {
        Ok(specs) => specs,
        Err(err) => {
            eprintln!("build_opspec: failed to parse op defs: {err}");
            std::process::exit(1);
        }
    };
    let total = specs.len();
    for (idx, spec) in specs.iter().enumerate() {
        let line = format!(
            "[{}/{}] -- generating opspec for {}[inplace={}, broadcast={}, accumulate={}]",
            idx + 1,
            total,
            spec.name,
            spec.inplace,
            spec.broadcast,
            spec.accumulate
        );
        print!("\r{:<120}", line);
        let _ = io::stdout().flush();
        thread::sleep(Duration::from_millis(200));
    }
    println!();
    if let Err(err) = openinfer_generator::op_schema::generate_cpu_kernels(&openinfer_dir) {
        eprintln!("build_opspec: failed to generate cpu kernels: {err}");
        std::process::exit(1);
    }
}

fn parse_opspecs(path: &Path) -> Result<Vec<OpSpecInfo>, String> {
    let contents = fs::read_to_string(path).map_err(|err| err.to_string())?;
    let file: File = syn::parse_file(&contents).map_err(|err| err.to_string())?;
    let ops_const = file
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Const(item) if item.ident == "OPS" => Some(item),
            _ => None,
        })
        .next()
        .ok_or_else(|| "missing OPS const in op_defs.rs".to_string())?;
    let expr = unwrap_reference(&ops_const.expr);
    let array = match expr {
        Expr::Array(array) => array,
        _ => return Err("OPS const is not an array literal".to_string()),
    };
    let mut specs = Vec::new();
    for elem in &array.elems {
        let expr = unwrap_reference(elem);
        let Expr::Struct(struct_expr) = expr else {
            continue;
        };
        let mut kind_ident = None;
        let mut inplace = false;
        let mut broadcast = false;
        let mut accumulate = false;
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
                "broadcast" => {
                    broadcast = extract_allow_flag(&field.expr)?;
                }
                "accumulate" => {
                    accumulate = extract_allow_flag(&field.expr)?;
                }
                _ => {}
            }
        }
        let kind_ident = kind_ident.ok_or_else(|| "missing kind ident".to_string())?;
        let name = op_kind_name(&kind_ident)?;
        specs.push(OpSpecInfo {
            name,
            inplace,
            broadcast,
            accumulate,
        });
    }
    Ok(specs)
}

fn unwrap_reference<'a>(expr: &'a Expr) -> &'a Expr {
    match expr {
        Expr::Reference(ExprReference { expr, .. }) => expr.as_ref(),
        other => other,
    }
}

fn extract_path_ident(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(ExprPath { path, .. }) => path.segments.last().map(|seg| seg.ident.to_string()),
        _ => None,
    }
}

fn extract_allow_flag(expr: &Expr) -> Result<bool, String> {
    let expr = unwrap_reference(expr);
    let ident = extract_path_ident(expr).ok_or_else(|| "expected path for support flag".to_string())?;
    match ident.as_str() {
        "Allow" => Ok(true),
        "Deny" => Ok(false),
        _ => Err(format!("unexpected support flag {ident}")),
    }
}

fn op_kind_name(kind: &str) -> Result<String, String> {
    let name = match kind {
        "Add" => "add",
        "Mul" => "mul",
        "Abs" => "abs",
        "Relu" => "relu",
        "Matmul" => "matmul",
        "IsFinite" => "is_finite",
        "Fill" => "fill",
        "Sub" => "sub",
        "Div" => "div",
        "FloorDiv" => "floor_div",
        "Rem" => "rem",
        "Fma" => "fma",
        "Neg" => "neg",
        "Sign" => "sign",
        "Recip" => "recip",
        "Min" => "min",
        "Max" => "max",
        "Clamp" => "clamp",
        "Floor" => "floor",
        "Ceil" => "ceil",
        "Round" => "round",
        "Trunc" => "trunc",
        "And" => "and",
        "Or" => "or",
        "Xor" => "xor",
        "Not" => "not",
        "Shl" => "shl",
        "Shr" => "shr",
        "Popcount" => "popcount",
        "Eq" => "eq",
        "Ne" => "ne",
        "Lt" => "lt",
        "Le" => "le",
        "Gt" => "gt",
        "Ge" => "ge",
        "Filter" => "filter",
        "IsNan" => "is_nan",
        "IsInf" => "is_inf",
        "IsNeg" => "is_neg",
        "SumAxis" => "sum_axis",
        "MeanAxis" => "mean_axis",
        "ProdAxis" => "prod_axis",
        "MaxAxis" => "max_axis",
        "MinAxis" => "min_axis",
        "ArgmaxAxis" => "argmax_axis",
        "ArgminAxis" => "argmin_axis",
        _ => return Err(format!("unknown OpKind::{kind}")),
    };
    Ok(name.to_string())
}
