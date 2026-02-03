use anyhow::Result;
use openinfer::{
    AttrValue, Device, Graph, MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs, OpKind,
    TensorValue,
};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

struct RoundingCase {
    op: OpKind,
    output: &'static str,
    attrs: OpAttrs,
}

#[test]
fn ops_rounding_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_rounding.oinf")?;
    let cases = [
        RoundingCase {
            op: OpKind::Floor,
            output: "floor_out",
            attrs: OpAttrs::none(),
        },
        RoundingCase {
            op: OpKind::Ceil,
            output: "ceil_out",
            attrs: OpAttrs::none(),
        },
        RoundingCase {
            op: OpKind::Round,
            output: "round_out",
            attrs: OpAttrs::none(),
        },
        RoundingCase {
            op: OpKind::Trunc,
            output: "trunc_out",
            attrs: OpAttrs::none(),
        },
        RoundingCase {
            op: OpKind::Sign,
            output: "sign_out",
            attrs: OpAttrs::none(),
        },
        RoundingCase {
            op: OpKind::Recip,
            output: "recip_out",
            attrs: div_by_zero_attrs(),
        },
    ];

    for device in common::test_targets() {
        for case in &cases {
            run_case(&model, device, case)?;
        }
    }
    Ok(())
}

fn run_case(model: &ModelLoader, device: Device, case: &RoundingCase) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let input = model.load_tensor("round_x")?;
    let expected = model.load_tensor(case.output)?;

    add_dynamic(&mut graph, "round_x", &input);
    add_volatile(&mut graph, case.output, &expected);

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: case.op,
            attrs: case.attrs.clone(),
            inputs: vec!["round_x".to_string()],
            output: case.output.to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping {:?} rounding on {:?}: {}", case.op, device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("round_x", input)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch(case.output)?;
    common::assert_tensor_close(&actual, &expected, device)?;
    Ok(())
}

fn div_by_zero_attrs() -> OpAttrs {
    OpAttrs {
        items: vec![OpAttr {
            name: "div_by_zero_mask".to_string(),
            value: AttrValue::Int(0),
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
