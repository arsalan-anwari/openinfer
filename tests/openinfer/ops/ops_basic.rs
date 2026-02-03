use anyhow::Result;
use openinfer::{
    AttrValue, Device, Graph, MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs, OpKind,
    TensorValue,
};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

struct OpCase {
    name: &'static str,
    op: OpKind,
    inputs: Vec<&'static str>,
    output: &'static str,
    attrs: OpAttrs,
}

#[test]
fn ops_basic_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_basic.oinf")?;
    let cases = vec![
        OpCase {
            name: "add",
            op: OpKind::Add,
            inputs: vec!["add_a", "add_b"],
            output: "add_out",
            attrs: OpAttrs::none(),
        },
        OpCase {
            name: "sub",
            op: OpKind::Sub,
            inputs: vec!["sub_a", "sub_b"],
            output: "sub_out",
            attrs: OpAttrs::none(),
        },
        OpCase {
            name: "mul",
            op: OpKind::Mul,
            inputs: vec!["mul_a", "mul_b"],
            output: "mul_out",
            attrs: OpAttrs::none(),
        },
        OpCase {
            name: "div",
            op: OpKind::Div,
            inputs: vec!["div_a", "div_b"],
            output: "div_out",
            attrs: op_attrs(OpKind::Div),
        },
        OpCase {
            name: "relu",
            op: OpKind::Relu,
            inputs: vec!["relu_x"],
            output: "relu_out",
            attrs: OpAttrs::none(),
        },
        OpCase {
            name: "abs",
            op: OpKind::Abs,
            inputs: vec!["abs_x"],
            output: "abs_out",
            attrs: OpAttrs::none(),
        },
        OpCase {
            name: "neg",
            op: OpKind::Neg,
            inputs: vec!["neg_x"],
            output: "neg_out",
            attrs: OpAttrs::none(),
        },
        OpCase {
            name: "min",
            op: OpKind::Min,
            inputs: vec!["min_a", "min_b"],
            output: "min_out",
            attrs: OpAttrs::none(),
        },
        OpCase {
            name: "max",
            op: OpKind::Max,
            inputs: vec!["max_a", "max_b"],
            output: "max_out",
            attrs: OpAttrs::none(),
        },
        OpCase {
            name: "clamp",
            op: OpKind::Clamp,
            inputs: vec!["clamp_x"],
            output: "clamp_out",
            attrs: op_attrs(OpKind::Clamp),
        },
        OpCase {
            name: "sum_axis",
            op: OpKind::SumAxis,
            inputs: vec!["sum_axis_x"],
            output: "sum_axis_out",
            attrs: op_attrs(OpKind::SumAxis),
        },
        OpCase {
            name: "mean_axis",
            op: OpKind::MeanAxis,
            inputs: vec!["mean_axis_x"],
            output: "mean_axis_out",
            attrs: op_attrs(OpKind::MeanAxis),
        },
    ];

    for device in common::test_targets() {
        for case in &cases {
            run_case(&model, device, case)?;
        }
    }
    Ok(())
}

fn run_case(model: &ModelLoader, device: Device, case: &OpCase) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let mut inputs = Vec::with_capacity(case.inputs.len());
    for &input_name in &case.inputs {
        let tensor = model.load_tensor(input_name)?;
        add_dynamic(&mut graph, input_name, &tensor);
        inputs.push((input_name.to_string(), tensor));
    }

    let expected = model.load_tensor(case.output)?;
    add_volatile(&mut graph, case.output, &expected);
    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: case.op,
            attrs: case.attrs.clone(),
            inputs: case.inputs.iter().map(|name| (*name).to_string()).collect(),
            output: case.output.to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!(
                    "Skipping {} on {:?}: {}",
                    case.name, device, err
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
    let actual: TensorValue = exec.fetch(case.output)?;
    common::assert_tensor_close(&actual, &expected, device)?;
    Ok(())
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

fn op_attrs(op: OpKind) -> OpAttrs {
    let mut items = Vec::new();
    match op {
        OpKind::Div | OpKind::FloorDiv | OpKind::Recip => {
            items.push(OpAttr {
                name: "div_by_zero_mask".to_string(),
                value: AttrValue::Int(0),
            });
        }
        OpKind::Clamp => {
            items.push(OpAttr {
                name: "min".to_string(),
                value: AttrValue::Float(0.0),
            });
            items.push(OpAttr {
                name: "max".to_string(),
                value: AttrValue::Float(3.0),
            });
        }
        OpKind::SumAxis | OpKind::MeanAxis => {
            items.push(OpAttr {
                name: "axes".to_string(),
                value: AttrValue::IntList(vec![1]),
            });
            items.push(OpAttr {
                name: "keepdims".to_string(),
                value: AttrValue::Bool(true),
            });
        }
        _ => {}
    }
    OpAttrs { items }
}
