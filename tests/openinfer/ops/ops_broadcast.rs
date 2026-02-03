use anyhow::Result;
use openinfer::{Device, Graph, MemoryKind, ModelLoader, NodeKind, OpAttrs, OpKind, TensorValue};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

#[test]
fn ops_broadcast_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_broadcast.oinf")?;

    for device in common::test_targets() {
        run_case(&model, device, OpKind::Add, "bcast_add_out")?;
        run_case(&model, device, OpKind::Mul, "bcast_mul_out")?;
    }
    Ok(())
}

fn run_case(model: &ModelLoader, device: Device, op: OpKind, output: &str) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let a = model.load_tensor("bcast_a")?;
    let b = model.load_tensor("bcast_b")?;
    let expected = model.load_tensor(output)?;

    add_dynamic(&mut graph, "bcast_a", &a);
    add_dynamic(&mut graph, "bcast_b", &b);
    add_volatile(&mut graph, output, &expected);

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op,
            attrs: OpAttrs::none(),
            inputs: vec!["bcast_a".to_string(), "bcast_b".to_string()],
            output: output.to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping {:?} broadcast on {:?}: {}", op, device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("bcast_a", a)?;
    exec.insert_dynamic("bcast_b", b)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch(output)?;
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
