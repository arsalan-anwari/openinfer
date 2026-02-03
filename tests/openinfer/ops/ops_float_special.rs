use anyhow::Result;
use openinfer::{op_schema, Device, DType, Graph, MemoryKind, ModelLoader, NodeKind, OpAttrs, OpKind, TensorValue};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

struct FloatCase {
    op: OpKind,
    input_a: &'static str,
    input_b: Option<&'static str>,
    output: &'static str,
    dtype: DType,
}

#[test]
fn ops_float_special_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_float_special.oinf")?;
    let cases = [
        FloatCase {
            op: OpKind::Add,
            input_a: "f16_a",
            input_b: Some("f16_b"),
            output: "f16_add_out",
            dtype: DType::F16,
        },
        FloatCase {
            op: OpKind::Relu,
            input_a: "f16_relu_x",
            input_b: None,
            output: "f16_relu_out",
            dtype: DType::F16,
        },
        FloatCase {
            op: OpKind::Add,
            input_a: "bf16_a",
            input_b: Some("bf16_b"),
            output: "bf16_add_out",
            dtype: DType::BF16,
        },
        FloatCase {
            op: OpKind::Relu,
            input_a: "bf16_relu_x",
            input_b: None,
            output: "bf16_relu_out",
            dtype: DType::BF16,
        },
        FloatCase {
            op: OpKind::Add,
            input_a: "f8_a",
            input_b: Some("f8_b"),
            output: "f8_add_out",
            dtype: DType::F8,
        },
        FloatCase {
            op: OpKind::Relu,
            input_a: "f8_relu_x",
            input_b: None,
            output: "f8_relu_out",
            dtype: DType::F8,
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

fn run_case(model: &ModelLoader, device: Device, case: &FloatCase) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let input_a = model.load_tensor(case.input_a)?;
    add_dynamic(&mut graph, case.input_a, &input_a);
    let mut inputs = vec![case.input_a.to_string()];
    if let Some(input_b_name) = case.input_b {
        let input_b = model.load_tensor(input_b_name)?;
        add_dynamic(&mut graph, input_b_name, &input_b);
        inputs.push(input_b_name.to_string());
    }

    let expected = model.load_tensor(case.output)?;
    add_volatile(&mut graph, case.output, &expected);

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: case.op,
            attrs: OpAttrs::none(),
            inputs,
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
    exec.insert_dynamic(case.input_a, input_a)?;
    if let Some(input_b_name) = case.input_b {
        let input_b = model.load_tensor(input_b_name)?;
        exec.insert_dynamic(input_b_name, input_b)?;
    }
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
