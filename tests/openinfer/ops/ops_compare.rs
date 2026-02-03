use anyhow::Result;
use openinfer::{Device, Graph, MemoryKind, ModelLoader, NodeKind, OpAttrs, OpKind, TensorValue};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

struct CompareCase {
    op: OpKind,
    output: &'static str,
}

#[test]
fn ops_compare_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_compare.oinf")?;
    let cases = [
        CompareCase {
            op: OpKind::Eq,
            output: "eq_out",
        },
        CompareCase {
            op: OpKind::Ne,
            output: "ne_out",
        },
        CompareCase {
            op: OpKind::Lt,
            output: "lt_out",
        },
        CompareCase {
            op: OpKind::Le,
            output: "le_out",
        },
        CompareCase {
            op: OpKind::Gt,
            output: "gt_out",
        },
        CompareCase {
            op: OpKind::Ge,
            output: "ge_out",
        },
    ];

    for device in common::test_targets() {
        for case in &cases {
            run_case(&model, device, case)?;
        }
    }
    Ok(())
}

fn run_case(model: &ModelLoader, device: Device, case: &CompareCase) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let a = model.load_tensor("cmp_a")?;
    let b = model.load_tensor("cmp_b")?;
    let expected = model.load_tensor(case.output)?;

    add_dynamic(&mut graph, "cmp_a", &a);
    add_dynamic(&mut graph, "cmp_b", &b);
    add_volatile(&mut graph, case.output, &expected);

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: case.op,
            attrs: OpAttrs::none(),
            inputs: vec!["cmp_a".to_string(), "cmp_b".to_string()],
            output: case.output.to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping {:?} compare on {:?}: {}", case.op, device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("cmp_a", a)?;
    exec.insert_dynamic("cmp_b", b)?;
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
