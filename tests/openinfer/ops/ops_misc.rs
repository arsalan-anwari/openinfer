use anyhow::Result;
use openinfer::{
    AttrValue, Device, Graph, MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs, OpKind,
    TensorValue,
};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

struct MiscCase {
    op: OpKind,
    output: &'static str,
    inputs: Vec<&'static str>,
    attrs: OpAttrs,
}

#[test]
fn ops_misc_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_misc.oinf")?;
    let cases = [
        MiscCase {
            op: OpKind::Fma,
            output: "fma_out",
            inputs: vec!["fma_a", "fma_b", "fma_c"],
            attrs: OpAttrs::none(),
        },
        MiscCase {
            op: OpKind::FloorDiv,
            output: "floor_div_out",
            inputs: vec!["floor_div_a", "floor_div_b"],
            attrs: div_by_zero_attrs(),
        },
        MiscCase {
            op: OpKind::Rem,
            output: "rem_out",
            inputs: vec!["rem_a", "rem_b"],
            attrs: OpAttrs::none(),
        },
        MiscCase {
            op: OpKind::IsNan,
            output: "is_nan_out",
            inputs: vec!["is_nan_x"],
            attrs: OpAttrs::none(),
        },
        MiscCase {
            op: OpKind::IsInf,
            output: "is_inf_out",
            inputs: vec!["is_inf_x"],
            attrs: OpAttrs::none(),
        },
        MiscCase {
            op: OpKind::IsNeg,
            output: "is_neg_out",
            inputs: vec!["is_neg_x"],
            attrs: OpAttrs::none(),
        },
        MiscCase {
            op: OpKind::IsFinite,
            output: "is_finite_out",
            inputs: vec!["is_finite_x"],
            attrs: OpAttrs::none(),
        },
    ];

    for device in common::test_targets() {
        for case in &cases {
            run_case(&model, device, case)?;
        }
    }
    Ok(())
}

fn run_case(model: &ModelLoader, device: Device, case: &MiscCase) -> Result<()> {
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
                eprintln!("Skipping {:?} misc on {:?}: {}", case.op, device, err);
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
