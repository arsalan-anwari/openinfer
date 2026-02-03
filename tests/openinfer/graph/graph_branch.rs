use anyhow::Result;
use openinfer::{Device, Graph, MemoryKind, ModelLoader, NodeKind, OpAttrs, OpKind, TensorValue};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

#[test]
fn graph_branch_parity() -> Result<()> {
    let model = common::load_baseline_model("graph/baseline/data/graph_branch.oinf")?;

    for device in common::test_targets() {
        run_graph_case(&model, device)?;
    }
    Ok(())
}

fn run_graph_case(model: &ModelLoader, device: Device) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let input = model.load_tensor("branch_x")?;
    let expected = model.load_tensor("branch_out")?;

    add_dynamic(&mut graph, "branch_x", &input);
    add_volatile(&mut graph, "branch_relu", &expected);
    add_volatile(&mut graph, "branch_abs", &expected);
    add_volatile(&mut graph, "branch_out", &expected);

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Relu,
            attrs: OpAttrs::none(),
            inputs: vec!["branch_x".to_string()],
            output: "branch_relu".to_string(),
        },
    )?;
    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Abs,
            attrs: OpAttrs::none(),
            inputs: vec!["branch_x".to_string()],
            output: "branch_abs".to_string(),
        },
    )?;
    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Add,
            attrs: OpAttrs::none(),
            inputs: vec!["branch_relu".to_string(), "branch_abs".to_string()],
            output: "branch_out".to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping graph_branch on {:?}: {}", device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("branch_x", input)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch("branch_out")?;
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
