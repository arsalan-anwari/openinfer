use anyhow::Result;
use openinfer::{
    AttrValue, Device, DType, Graph, MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs, OpKind,
    TensorValue,
};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

struct CastCase {
    input: &'static str,
    output: &'static str,
    to: DType,
}

#[test]
fn ops_cast_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_cast.oinf")?;
    let cases = [
        CastCase {
            input: "cast_f32",
            output: "cast_f32_to_i32",
            to: DType::I32,
        },
        CastCase {
            input: "cast_f32",
            output: "cast_f32_to_u8",
            to: DType::U8,
        },
        CastCase {
            input: "cast_i32",
            output: "cast_i32_to_f32",
            to: DType::F32,
        },
    ];

    for device in common::test_targets() {
        for case in &cases {
            run_case(&model, device, case)?;
        }
    }
    Ok(())
}

fn run_case(model: &ModelLoader, device: Device, case: &CastCase) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let input = model.load_tensor(case.input)?;
    let expected = model.load_tensor(case.output)?;

    add_dynamic(&mut graph, case.input, &input);
    add_volatile(&mut graph, case.output, &expected);

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Cast,
            attrs: cast_attrs(case.to),
            inputs: vec![case.input.to_string()],
            output: case.output.to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping cast to {:?} on {:?}: {}", case.to, device, err);
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

fn cast_attrs(to: DType) -> OpAttrs {
    OpAttrs {
        items: vec![OpAttr {
            name: "to".to_string(),
            value: AttrValue::DType(to),
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
