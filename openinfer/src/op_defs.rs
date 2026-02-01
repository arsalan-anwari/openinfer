use std::collections::HashMap;

use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;
use serde::Deserialize;

use crate::graph::{AttrValue, OpAttrs, OpKind};
use crate::tensor::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum OpAttrType {
    Scalar,
    DType,
    Tensor,
    String,
    IntList,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ScalarAttrKind {
    Float,
    Int,
    UInt,
    Bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpAttrDef {
    pub name: &'static str,
    pub kind: OpAttrType,
    pub scalar_kinds: &'static [ScalarAttrKind],
}

impl OpAttrDef {
    pub const fn new(name: &'static str, kind: OpAttrType) -> Self {
        Self {
            name,
            kind,
            scalar_kinds: &[],
        }
    }

    pub const fn scalar(name: &'static str, scalar_kinds: &'static [ScalarAttrKind]) -> Self {
        Self {
            name,
            kind: OpAttrType::Scalar,
            scalar_kinds,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct OpDTypeSupport {
    pub normal: &'static [DType],
    pub accumulate: &'static [(DType, DType)],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum BroadcastSupport {
    Deny,
    Allow,
}

impl BroadcastSupport {
    pub fn allow(self) -> bool {
        matches!(self, BroadcastSupport::Allow)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum InplaceSupport {
    Deny,
    Allow,
}

impl InplaceSupport {
    pub fn allow(self) -> bool {
        matches!(self, InplaceSupport::Allow)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AccumulateSupport {
    Deny,
    Allow,
}

impl AccumulateSupport {
    pub fn allow(self) -> bool {
        matches!(self, AccumulateSupport::Allow)
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct OpSchema {
    pub kind: OpKind,
    pub inputs: InputArity,
    pub outputs: OutputArity,
    pub attrs: &'static [OpAttrDef],
    pub broadcast: BroadcastSupport,
    pub inplace: InplaceSupport,
    pub accumulate: AccumulateSupport,
    pub type_rule: TypeRule,
    pub dtype_support: Option<&'static OpDTypeSupport>,
    pub output_dtypes: Option<&'static [DType]>,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum TypeRule {
    SameAsInput(usize),
    Fixed(DType),
    AccFromAttr { attr: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum InputArity {
    Fixed(usize),
    AtLeast(usize),
    Any,
}

impl InputArity {
    pub fn allows(self, count: usize) -> bool {
        match self {
            InputArity::Fixed(expected) => count == expected,
            InputArity::AtLeast(min) => count >= min,
            InputArity::Any => true,
        }
    }

    pub fn fixed(self) -> Option<usize> {
        match self {
            InputArity::Fixed(count) => Some(count),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum OutputArity {
    Fixed(usize),
    AtLeast(usize),
    Any,
}

#[allow(dead_code)]
impl OutputArity {
    pub fn allows(self, count: usize) -> bool {
        match self {
            OutputArity::Fixed(expected) => count == expected,
            OutputArity::AtLeast(min) => count >= min,
            OutputArity::Any => true,
        }
    }

    #[allow(dead_code)]
    pub fn fixed(self) -> Option<usize> {
        match self {
            OutputArity::Fixed(count) => Some(count),
            _ => None,
        }
    }
}

impl TypeRule {
    pub fn output_dtype(self, inputs: &[DType], attrs: &OpAttrs) -> Result<DType> {
        match self {
            TypeRule::SameAsInput(index) => inputs
                .get(index)
                .copied()
                .ok_or_else(|| anyhow!("missing input dtype at {}", index)),
            TypeRule::Fixed(dtype) => Ok(dtype),
            TypeRule::AccFromAttr { attr } => attrs
                .items
                .iter()
                .find(|item| item.name == attr)
                .ok_or_else(|| anyhow!("missing {} attribute", attr))
                .and_then(|item| match &item.value {
                    AttrValue::DType(dtype) => Ok(*dtype),
                    _ => Err(anyhow!("{} attribute must be a dtype", attr)),
                }),
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct OpRegistry {
    schemas: Vec<OpSchema>,
    dtype_supports: HashMap<String, &'static OpDTypeSupport>,
    output_dtype_sets: HashMap<String, &'static [DType]>,
}

static REGISTRY: OnceCell<OpRegistry> = OnceCell::new();

#[derive(Debug, Deserialize)]
struct OpsFile {
    version: u32,
    attr_defs: HashMap<String, AttrDefJson>,
    dtype_sets: HashMap<String, DTypeSupportJson>,
    output_dtype_sets: Option<HashMap<String, Vec<String>>>,
    ops: Vec<OpSchemaJson>,
}

#[derive(Debug, Deserialize)]
struct AttrDefJson {
    kind: String,
    #[serde(default)]
    scalar_kinds: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct OpSchemaJson {
    name: String,
    kind: OpKind,
    inputs: ArityJson,
    outputs: ArityJson,
    #[serde(default)]
    attrs: Vec<String>,
    broadcast: String,
    inplace: String,
    accumulate: String,
    type_rule: TypeRuleJson,
    dtype_support_ref: Option<String>,
    output_dtypes_ref: Option<String>,
    #[serde(default)]
    devices: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ArityJson {
    arity: String,
    count: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct TypeRuleJson {
    kind: String,
    index: Option<usize>,
    dtype: Option<String>,
    attr: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DTypeSupportJson {
    normal: Vec<String>,
    #[serde(default)]
    accumulate: Vec<AccumulatePairJson>,
}

#[derive(Debug, Deserialize)]
struct AccumulatePairJson {
    input: String,
    acc: String,
}

fn registry() -> &'static OpRegistry {
    REGISTRY.get_or_init(|| {
        load_registry().unwrap_or_else(|err| panic!("ops registry init failed: {err}"))
    })
}

fn load_registry() -> Result<OpRegistry> {
    let json = include_str!("../../ops.json");
    let file: OpsFile = serde_json::from_str(json)?;
    if file.version != 1 {
        return Err(anyhow!("unsupported ops.json version {}", file.version));
    }

    let attr_defs = build_attr_defs(&file.attr_defs)?;
    let dtype_supports = build_dtype_supports(&file.dtype_sets)?;
    let output_dtype_sets = build_output_dtype_sets(file.output_dtype_sets.as_ref())?;
    let mut schemas = Vec::with_capacity(file.ops.len());
    for op in file.ops {
        let attrs = build_attr_list(&attr_defs, &op.attrs)?;
        let inputs = parse_input_arity(&op.inputs)?;
        let outputs = parse_output_arity(&op.outputs)?;
        let broadcast = parse_broadcast(&op.broadcast)?;
        let inplace = parse_inplace(&op.inplace)?;
        let accumulate = parse_accumulate(&op.accumulate)?;
        let type_rule = parse_type_rule(op.type_rule)?;
        let dtype_support = op
            .dtype_support_ref
            .as_deref()
            .and_then(|name| dtype_supports.get(name).copied())
            .ok_or_else(|| anyhow!("unknown dtype_support_ref for {}", op.name))?;
        let output_dtypes = match op.output_dtypes_ref.as_deref() {
            Some(name) => Some(
                output_dtype_sets
                    .get(name)
                    .copied()
                    .ok_or_else(|| anyhow!("unknown output_dtypes_ref for {}", op.name))?,
            ),
            None => None,
        };
        schemas.push(OpSchema {
            kind: op.kind,
            inputs,
            outputs,
            attrs,
            broadcast,
            inplace,
            accumulate,
            type_rule,
            dtype_support: Some(dtype_support),
            output_dtypes,
        });
    }
    Ok(OpRegistry {
        schemas,
        dtype_supports,
        output_dtype_sets,
    })
}

fn build_attr_defs(defs: &HashMap<String, AttrDefJson>) -> Result<HashMap<String, OpAttrDef>> {
    let mut out = HashMap::new();
    for (name, def) in defs {
        let name_static: &'static str = Box::leak(name.clone().into_boxed_str());
        let kind = match def.kind.as_str() {
            "scalar" => OpAttrType::Scalar,
            "dtype" => OpAttrType::DType,
            "tensor" => OpAttrType::Tensor,
            "string" => OpAttrType::String,
            "int_list" => OpAttrType::IntList,
            other => return Err(anyhow!("unknown attr kind {other} for {name}")),
        };
        let scalar_kinds = if matches!(kind, OpAttrType::Scalar) {
            let kinds = def
                .scalar_kinds
                .iter()
                .map(|kind| match kind.as_str() {
                    "float" => Ok(ScalarAttrKind::Float),
                    "int" => Ok(ScalarAttrKind::Int),
                    "uint" => Ok(ScalarAttrKind::UInt),
                    "bool" => Ok(ScalarAttrKind::Bool),
                    other => Err(anyhow!("unknown scalar kind {other} for {name}")),
                })
                .collect::<Result<Vec<_>>>()?;
            Box::leak(kinds.into_boxed_slice()) as &'static [ScalarAttrKind]
        } else {
            &[]
        };
        out.insert(
            name.clone(),
            OpAttrDef {
                name: name_static,
                kind,
                scalar_kinds,
            },
        );
    }
    Ok(out)
}

fn build_attr_list(
    defs: &HashMap<String, OpAttrDef>,
    attrs: &[String],
) -> Result<&'static [OpAttrDef]> {
    let mut out = Vec::with_capacity(attrs.len());
    for attr in attrs {
        let def = defs
            .get(attr)
            .copied()
            .ok_or_else(|| anyhow!("unknown attr {attr} in ops.json"))?;
        out.push(def);
    }
    Ok(Box::leak(out.into_boxed_slice()))
}

fn parse_input_arity(arity: &ArityJson) -> Result<InputArity> {
    match arity.arity.as_str() {
        "fixed" => Ok(InputArity::Fixed(required_count(arity, "fixed")?)),
        "at_least" => Ok(InputArity::AtLeast(required_count(arity, "at_least")?)),
        "any" => Ok(InputArity::Any),
        other => Err(anyhow!("unknown input arity {other}")),
    }
}

fn parse_output_arity(arity: &ArityJson) -> Result<OutputArity> {
    match arity.arity.as_str() {
        "fixed" => Ok(OutputArity::Fixed(required_count(arity, "fixed")?)),
        "at_least" => Ok(OutputArity::AtLeast(required_count(arity, "at_least")?)),
        "any" => Ok(OutputArity::Any),
        other => Err(anyhow!("unknown output arity {other}")),
    }
}

fn required_count(arity: &ArityJson, label: &str) -> Result<usize> {
    arity
        .count
        .ok_or_else(|| anyhow!("missing count for {label} arity"))
}

fn parse_broadcast(value: &str) -> Result<BroadcastSupport> {
    match value {
        "allow" => Ok(BroadcastSupport::Allow),
        "deny" => Ok(BroadcastSupport::Deny),
        other => Err(anyhow!("unknown broadcast support {other}")),
    }
}

fn parse_inplace(value: &str) -> Result<InplaceSupport> {
    match value {
        "allow" => Ok(InplaceSupport::Allow),
        "deny" => Ok(InplaceSupport::Deny),
        other => Err(anyhow!("unknown inplace support {other}")),
    }
}

fn parse_accumulate(value: &str) -> Result<AccumulateSupport> {
    match value {
        "allow" => Ok(AccumulateSupport::Allow),
        "deny" => Ok(AccumulateSupport::Deny),
        other => Err(anyhow!("unknown accumulate support {other}")),
    }
}

fn parse_type_rule(rule: TypeRuleJson) -> Result<TypeRule> {
    match rule.kind.as_str() {
        "same_as_input" => Ok(TypeRule::SameAsInput(
            rule.index.ok_or_else(|| anyhow!("missing index for same_as_input"))?,
        )),
        "fixed" => {
            let dtype = rule
                .dtype
                .ok_or_else(|| anyhow!("missing dtype for fixed type_rule"))?;
            Ok(TypeRule::Fixed(DType::from_ident(&dtype)?))
        }
        "acc_from_attr" => {
            let attr = rule
                .attr
                .ok_or_else(|| anyhow!("missing attr for acc_from_attr"))?;
            let attr_static: &'static str = Box::leak(attr.into_boxed_str());
            Ok(TypeRule::AccFromAttr { attr: attr_static })
        }
        other => Err(anyhow!("unknown type_rule {other}")),
    }
}

fn build_dtype_supports(
    dtype_sets: &HashMap<String, DTypeSupportJson>,
) -> Result<HashMap<String, &'static OpDTypeSupport>> {
    let mut out = HashMap::new();
    for (name, support) in dtype_sets {
        let normal = support
            .normal
            .iter()
            .map(|ident| DType::from_ident(ident))
            .collect::<Result<Vec<_>>>()?;
        let accumulate = support
            .accumulate
            .iter()
            .map(|pair| {
                Ok((
                    DType::from_ident(&pair.input)?,
                    DType::from_ident(&pair.acc)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let normal_static = Box::leak(normal.into_boxed_slice());
        let acc_static = Box::leak(accumulate.into_boxed_slice());
        let support_static: &'static OpDTypeSupport = Box::leak(Box::new(OpDTypeSupport {
            normal: normal_static,
            accumulate: acc_static,
        }));
        out.insert(name.clone(), support_static);
    }
    Ok(out)
}

fn build_output_dtype_sets(
    output_sets: Option<&HashMap<String, Vec<String>>>,
) -> Result<HashMap<String, &'static [DType]>> {
    let mut out = HashMap::new();
    if let Some(output_sets) = output_sets {
        for (name, dtypes) in output_sets {
            let converted = dtypes
                .iter()
                .map(|ident| DType::from_ident(ident))
                .collect::<Result<Vec<_>>>()?;
            let leaked: &'static [DType] = Box::leak(converted.into_boxed_slice());
            out.insert(name.clone(), leaked);
        }
    }
    Ok(out)
}

#[allow(unused)]
pub fn acc_dtype(attrs: &OpAttrs) -> Result<DType> {
    attrs
        .items
        .iter()
        .find(|attr| attr.name == "acc")
        .ok_or_else(|| anyhow!("missing acc attribute"))
        .and_then(|attr| match &attr.value {
            AttrValue::DType(dtype) => Ok(*dtype),
            _ => Err(anyhow!("acc attribute must be a dtype")),
        })
}

pub fn op_schema(kind: OpKind) -> Option<&'static OpSchema> {
    registry().schemas.iter().find(|op| op.kind == kind)
}

pub fn init_ops_registry() {
    let _ = registry();
}
