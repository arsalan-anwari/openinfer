use anyhow::Result;
use openinfer::{
    op_schema, AttrValue, Device, DType, Graph, MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs,
    OpKind, TensorValue,
};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

#[test]
fn ops_accumulate_inplace_parity() -> Result<()> {
    let model =
        common::load_baseline_model("ops/baseline/data/ops_accumulate_inplace.oinf")?;

    for device in common::test_targets() {
        run_accumulate_case(&model, device)?;
        run_inplace_case(&model, device)?;
    }
    Ok(())
}

fn run_accumulate_case(model: &ModelLoader, device: Device) -> Result<()> {
    let schema = match op_schema(OpKind::Add) {
        Some(schema) => schema,
        None => return Ok(()),
    };
    let Some(support) = schema.dtype_support else {
        return Ok(());
    };
    let target_pair = support
        .accumulate
        .iter()
        .find(|&&(in_dtype, acc_dtype)| in_dtype == DType::F16 && acc_dtype == DType::F32);
    if target_pair.is_none() {
        eprintln!("Skipping accumulate add: no F16->F32 support");
        return Ok(());
    }

    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let a = model.load_tensor("acc_a")?;
    let b = model.load_tensor("acc_b")?;
    let expected = model.load_tensor("acc_out")?;

    add_dynamic(&mut graph, "acc_a", &a);
    add_dynamic(&mut graph, "acc_b", &b);
    add_volatile(&mut graph, "acc_out", &expected);

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Add,
            attrs: acc_attrs(DType::F32),
            inputs: vec!["acc_a".to_string(), "acc_b".to_string()],
            output: "acc_out".to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping accumulate add on {:?}: {}", device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("acc_a", a)?;
    exec.insert_dynamic("acc_b", b)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch("acc_out")?;
    common::assert_tensor_close(&actual, &expected, device)?;
    Ok(())
}

fn run_inplace_case(model: &ModelLoader, device: Device) -> Result<()> {
    let schema = match op_schema(OpKind::Add) {
        Some(schema) => schema,
        None => return Ok(()),
    };
    if !schema.inplace.allow() {
        eprintln!("Skipping inplace add: not supported");
        return Ok(());
    }

    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let a = model.load_tensor("inplace_a")?;
    let b = model.load_tensor("inplace_b")?;
    let expected = model.load_tensor("inplace_out")?;

    add_dynamic(&mut graph, "inplace_a", &a);
    add_dynamic(&mut graph, "inplace_b", &b);

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Add,
            attrs: OpAttrs::none(),
            inputs: vec!["inplace_a".to_string(), "inplace_b".to_string()],
            output: "inplace_a".to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping inplace add on {:?}: {}", device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("inplace_a", a)?;
    exec.insert_dynamic("inplace_b", b)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch("inplace_a")?;
    common::assert_tensor_close(&actual, &expected, device)?;
    Ok(())
}

fn acc_attrs(acc: DType) -> OpAttrs {
    OpAttrs {
        items: vec![OpAttr {
            name: "acc".to_string(),
            value: AttrValue::DType(acc),
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
