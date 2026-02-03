use anyhow::Result;
use openinfer::{AttrValue, Device, Graph, MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs, OpKind, TensorValue};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

struct BitwiseCase {
    op: OpKind,
    output: &'static str,
    attrs: OpAttrs,
    inputs: Vec<&'static str>,
}

#[test]
fn ops_bitwise_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_bitwise.oinf")?;
    let cases = [
        BitwiseCase {
            op: OpKind::And,
            output: "and_out",
            attrs: OpAttrs::none(),
            inputs: vec!["bit_a", "bit_b"],
        },
        BitwiseCase {
            op: OpKind::Or,
            output: "or_out",
            attrs: OpAttrs::none(),
            inputs: vec!["bit_a", "bit_b"],
        },
        BitwiseCase {
            op: OpKind::Xor,
            output: "xor_out",
            attrs: OpAttrs::none(),
            inputs: vec!["bit_a", "bit_b"],
        },
        BitwiseCase {
            op: OpKind::Not,
            output: "not_out",
            attrs: OpAttrs::none(),
            inputs: vec!["bit_a"],
        },
        BitwiseCase {
            op: OpKind::Shl,
            output: "shl_out",
            attrs: shift_attrs(),
            inputs: vec!["bit_a"],
        },
        BitwiseCase {
            op: OpKind::Shr,
            output: "shr_out",
            attrs: shift_attrs(),
            inputs: vec!["bit_a"],
        },
        BitwiseCase {
            op: OpKind::Popcount,
            output: "popcount_out",
            attrs: OpAttrs::none(),
            inputs: vec!["bit_a"],
        },
    ];

    for device in common::test_targets() {
        for case in &cases {
            run_case(&model, device, case)?;
        }
    }
    Ok(())
}

fn run_case(model: &ModelLoader, device: Device, case: &BitwiseCase) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let mut inputs = Vec::new();
    for &name in &case.inputs {
        let tensor = model.load_tensor(name)?;
        add_dynamic(&mut graph, name, &tensor);
        inputs.push((name.to_string(), tensor));
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
                eprintln!("Skipping {:?} bitwise on {:?}: {}", case.op, device, err);
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

fn shift_attrs() -> OpAttrs {
    OpAttrs {
        items: vec![OpAttr {
            name: "bits".to_string(),
            value: AttrValue::Int(1),
        }],
    }
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
