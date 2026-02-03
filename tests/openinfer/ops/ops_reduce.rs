use anyhow::Result;
use openinfer::{AttrValue, Device, Graph, MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs, OpKind, TensorValue};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

struct ReduceCase {
    op: OpKind,
    output: &'static str,
    attrs: OpAttrs,
}

#[test]
fn ops_reduce_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_reduce.oinf")?;
    let cases = [
        ReduceCase {
            op: OpKind::SumAxis,
            output: "sum_axis_out",
            attrs: reduce_attrs(),
        },
        ReduceCase {
            op: OpKind::MeanAxis,
            output: "mean_axis_out",
            attrs: reduce_attrs(),
        },
        ReduceCase {
            op: OpKind::ProdAxis,
            output: "prod_axis_out",
            attrs: reduce_attrs(),
        },
        ReduceCase {
            op: OpKind::MinAxis,
            output: "min_axis_out",
            attrs: reduce_attrs(),
        },
        ReduceCase {
            op: OpKind::MaxAxis,
            output: "max_axis_out",
            attrs: reduce_attrs(),
        },
        ReduceCase {
            op: OpKind::ArgmaxAxis,
            output: "argmax_axis_out",
            attrs: arg_reduce_attrs(),
        },
        ReduceCase {
            op: OpKind::ArgminAxis,
            output: "argmin_axis_out",
            attrs: arg_reduce_attrs(),
        },
    ];

    for device in common::test_targets() {
        for case in &cases {
            run_case(&model, device, case)?;
        }
    }
    Ok(())
}

fn run_case(model: &ModelLoader, device: Device, case: &ReduceCase) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let input = model.load_tensor("reduce_x")?;
    let expected = model.load_tensor(case.output)?;

    add_dynamic(&mut graph, "reduce_x", &input);
    add_volatile(&mut graph, case.output, &expected);

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: case.op,
            attrs: case.attrs.clone(),
            inputs: vec!["reduce_x".to_string()],
            output: case.output.to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping {:?} reduce on {:?}: {}", case.op, device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("reduce_x", input)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch(case.output)?;
    common::assert_tensor_close(&actual, &expected, device)?;
    Ok(())
}

fn reduce_attrs() -> OpAttrs {
    OpAttrs {
        items: vec![
            OpAttr {
                name: "axes".to_string(),
                value: AttrValue::IntList(vec![1]),
            },
            OpAttr {
                name: "keepdims".to_string(),
                value: AttrValue::Bool(true),
            },
        ],
    }
}

fn arg_reduce_attrs() -> OpAttrs {
    OpAttrs {
        items: vec![
            OpAttr {
                name: "axis".to_string(),
                value: AttrValue::Int(1),
            },
            OpAttr {
                name: "keepdims".to_string(),
                value: AttrValue::Bool(true),
            },
            OpAttr {
                name: "select_first".to_string(),
                value: AttrValue::Bool(true),
            },
        ],
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
