use anyhow::Result;
use openinfer::{Device, Graph, MemoryKind, ModelLoader, NodeKind, OpAttrs, OpKind, TensorValue};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

#[test]
fn graph_simple_parity() -> Result<()> {
    let model = common::load_baseline_model("graph/baseline/data/graph_simple.oinf")?;

    for device in common::test_targets() {
        run_graph_case(&model, device)?;
    }
    Ok(())
}

fn run_graph_case(model: &ModelLoader, device: Device) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let input = model.load_tensor("simple_x")?;
    let weights = model.load_tensor("simple_w")?;
    let bias = model.load_tensor("simple_b")?;
    let expected = model.load_tensor("simple_out")?;

    add_dynamic(&mut graph, "simple_x", &input);
    add_dynamic(&mut graph, "simple_w", &weights);
    add_dynamic(&mut graph, "simple_b", &bias);
    add_volatile(&mut graph, "simple_matmul", &expected);
    add_volatile(&mut graph, "simple_add", &expected);
    add_volatile(&mut graph, "simple_out", &expected);

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Matmul,
            attrs: OpAttrs::none(),
            inputs: vec!["simple_x".to_string(), "simple_w".to_string()],
            output: "simple_matmul".to_string(),
        },
    )?;
    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Add,
            attrs: OpAttrs::none(),
            inputs: vec!["simple_matmul".to_string(), "simple_b".to_string()],
            output: "simple_add".to_string(),
        },
    )?;
    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Relu,
            attrs: OpAttrs::none(),
            inputs: vec!["simple_add".to_string()],
            output: "simple_out".to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping graph_simple on {:?}: {}", device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("simple_x", input)?;
    exec.insert_dynamic("simple_w", weights)?;
    exec.insert_dynamic("simple_b", bias)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch("simple_out")?;
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
