use anyhow::Result;
use openinfer::{AttrValue, DType, Device, Graph, MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs, OpKind, TensorValue};
use serde::Deserialize;
use std::collections::HashMap;

use crate::common;

const ENTRY_BLOCK: &str = "entry";

#[derive(Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    file: String,
    name: String,
    op: String,
    mode: String,
    inputs: Vec<String>,
    output_var: String,
    expected: String,
    attrs: Vec<AttrSpec>,
}

#[derive(Deserialize)]
struct AttrSpec {
    name: String,
    kind: String,
    #[serde(default)]
    scalar_kind: Option<String>,
    value: serde_json::Value,
}

#[test]
fn ops_full_matrix_parity() -> Result<()> {
    let manifest_path =
        common::baseline_path("ops/baseline/data/full_matrix/manifest.json");
    let manifest: Manifest =
        serde_json::from_str(&std::fs::read_to_string(manifest_path)?)?;
    let mut models: HashMap<String, ModelLoader> = HashMap::new();

    for device in common::test_targets() {
        for case in &manifest.cases {
            run_case(&mut models, device, case)?;
        }
    }
    Ok(())
}

fn run_case(
    models: &mut HashMap<String, ModelLoader>,
    device: Device,
    case: &Case,
) -> Result<()> {
    let model = models.entry(case.file.clone()).or_insert_with(|| {
        common::load_baseline_model(&format!("ops/baseline/data/{}", case.file))
            .expect("load baseline model")
    });

    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let mut inputs = Vec::with_capacity(case.inputs.len());
    for input_name in &case.inputs {
        let tensor = model.load_tensor(input_name)?;
        add_dynamic(&mut graph, input_name, &tensor);
        inputs.push((input_name.clone(), tensor));
    }

    let expected = model.load_tensor(&case.expected)?;
    if !case.inputs.contains(&case.output_var) {
        add_volatile(&mut graph, &case.output_var, &expected);
    }

    let op_kind = case
        .op
        .parse::<OpKind>()
        .map_err(|err| anyhow::anyhow!("unknown op {}: {}", case.op, err))?;
    let attrs = build_attrs(&case.attrs)?;

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: op_kind,
            attrs,
            inputs: case.inputs.clone(),
            output: case.output_var.clone(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!(
                    "Skipping {} (mode {}) on {:?}: {}",
                    case.name, case.mode, device, err
                );
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    for (name, tensor) in inputs {
        exec.insert_dynamic(&name, tensor)?;
    }
    exec.step()?;
    let actual: TensorValue = exec.fetch(&case.output_var)?;
    common::assert_tensor_close(&actual, &expected, device).map_err(|err| {
        anyhow::anyhow!(
            "case {} (op {}, mode {}, input {:?}, output {}): {}",
            case.name,
            case.op,
            case.mode,
            case.inputs,
            case.output_var,
            err
        )
    })?;
    Ok(())
}

fn build_attrs(attrs: &[AttrSpec]) -> Result<OpAttrs> {
    let mut items = Vec::new();
    for attr in attrs {
        let value = match attr.kind.as_str() {
            "dtype" => {
                let dtype = attr
                    .value
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("dtype value must be string"))?;
                AttrValue::DType(DType::from_ident(dtype)?)
            }
            "string" => AttrValue::Str(
                attr.value
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("string value must be string"))?
                    .to_string(),
            ),
            "int_list" => {
                let items = attr
                    .value
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("int_list must be array"))?;
                let mut out = Vec::new();
                for item in items {
                    out.push(
                        item.as_i64()
                            .ok_or_else(|| anyhow::anyhow!("int_list item must be int"))?,
                    );
                }
                AttrValue::IntList(out)
            }
            "scalar" => match attr.scalar_kind.as_deref() {
                Some("int") => AttrValue::Int(
                    attr.value
                        .as_i64()
                        .ok_or_else(|| anyhow::anyhow!("int scalar must be int"))?,
                ),
                Some("uint") => AttrValue::UInt(
                    attr.value
                        .as_u64()
                        .ok_or_else(|| anyhow::anyhow!("uint scalar must be uint"))?,
                ),
                Some("float") => AttrValue::Float(
                    attr.value
                        .as_f64()
                        .ok_or_else(|| anyhow::anyhow!("float scalar must be number"))?
                        as f32,
                ),
                Some("bool") => AttrValue::Bool(
                    attr.value
                        .as_bool()
                        .ok_or_else(|| anyhow::anyhow!("bool scalar must be bool"))?,
                ),
                other => {
                    return Err(anyhow::anyhow!(
                        "unsupported scalar kind {:?}",
                        other
                    ))
                }
            },
            other => {
                return Err(anyhow::anyhow!("unsupported attr kind {}", other));
            }
        };
        items.push(OpAttr {
            name: attr.name.clone(),
            value,
        });
    }
    Ok(OpAttrs { items })
}

fn add_dynamic(graph: &mut Graph, name: &str, tensor: &TensorValue) {
    graph.add_var(
        MemoryKind::Dynamic,
        name,
        tensor.dtype(),
        dims_from_shape(tensor.shape()),
        None,
        None,
        Vec::new(),
        None,
        false,
        Vec::new(),
        Vec::new(),
    );
}

fn add_volatile(graph: &mut Graph, name: &str, tensor: &TensorValue) {
    graph.add_var(
        MemoryKind::Volatile,
        name,
        tensor.dtype(),
        dims_from_shape(tensor.shape()),
        None,
        None,
        Vec::new(),
        None,
        false,
        Vec::new(),
        Vec::new(),
    );
}

fn dims_from_shape(shape: &[usize]) -> Vec<String> {
    shape.iter().map(|dim| dim.to_string()).collect()
}
