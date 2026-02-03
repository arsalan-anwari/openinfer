use anyhow::Result;
use openinfer::{Device, Graph, MemoryKind, ModelLoader, NodeKind, OpAttrs, OpKind, TensorValue};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

#[test]
fn ops_matmul_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_matmul.oinf")?;

    for device in common::test_targets() {
        run_matmul_case(&model, device)?;
    }
    Ok(())
}

fn run_matmul_case(model: &ModelLoader, device: Device) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let left = model.load_tensor("matmul_a")?;
    let right = model.load_tensor("matmul_b")?;
    let expected = model.load_tensor("matmul_out")?;

    add_dynamic(&mut graph, "matmul_a", &left);
    add_dynamic(&mut graph, "matmul_b", &right);
    add_volatile(&mut graph, "matmul_out", &expected);

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Matmul,
            attrs: OpAttrs::none(),
            inputs: vec!["matmul_a".to_string(), "matmul_b".to_string()],
            output: "matmul_out".to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping matmul on {:?}: {}", device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("matmul_a", left)?;
    exec.insert_dynamic("matmul_b", right)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch("matmul_out")?;
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
