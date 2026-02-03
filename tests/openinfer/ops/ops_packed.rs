use anyhow::Result;
use openinfer::{op_schema, AttrValue, Device, DType, Graph, MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs, OpKind, TensorValue};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

struct PackedCase {
    op: OpKind,
    input: &'static str,
    output: &'static str,
    dtype: DType,
}

#[test]
fn ops_packed_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_packed.oinf")?;
    let cases = [
        PackedCase {
            op: OpKind::ArgmaxAxis,
            input: "packed_i4_x",
            output: "packed_i4_argmax_out",
            dtype: DType::I4,
        },
        PackedCase {
            op: OpKind::ArgminAxis,
            input: "packed_u2_x",
            output: "packed_u2_argmin_out",
            dtype: DType::U2,
        },
    ];

    for device in common::test_targets() {
        for case in &cases {
            if !op_supports_dtype(case.op, case.dtype) {
                eprintln!("Skipping {:?} for {:?}", case.op, case.dtype);
                continue;
            }
            run_case(&model, device, case)?;
        }
    }
    Ok(())
}

fn op_supports_dtype(op: OpKind, dtype: DType) -> bool {
    let Some(schema) = op_schema(op) else {
        return false;
    };
    let Some(support) = schema.dtype_support else {
        return false;
    };
    support.normal.iter().any(|&d| d == dtype)
}

fn run_case(model: &ModelLoader, device: Device, case: &PackedCase) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let input = model.load_tensor(case.input)?;
    add_dynamic(&mut graph, case.input, &input);

    let expected = model.load_tensor(case.output)?;
    add_volatile(&mut graph, case.output, &expected);

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: case.op,
            attrs: arg_reduce_attrs(),
            inputs: vec![case.input.to_string()],
            output: case.output.to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping {:?} on {:?}: {}", case.op, device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic(case.input, input)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch(case.output)?;
    common::assert_tensor_close(&actual, &expected, device)?;
    Ok(())
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
