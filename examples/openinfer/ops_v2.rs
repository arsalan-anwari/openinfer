use anyhow::Result;
use openinfer::{
    op_schema, AttrValue, BF16, DType, Device, Executor, F16, F8, Graph,
    MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs, OpKind, Simulator, TensorValue,
    TypeRule,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

mod util;
use util::select_device;

const ENTRY_BLOCK: &str = "entry";

#[derive(Debug, Clone)]
struct DynamicVar {
    name: String,
    dtype: DType,
    dims: Vec<String>,
}

fn main() -> Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/ops_v2_model.oinf");
    let model = ModelLoader::open(model_path)?;
    let (graph, dynamic_vars) = build_graph()?;

    let device = select_device()?;
    openinfer::log!("⚠️ == Pass but drift. ✅ == Pass with no drift. ❌ == Fail");
    openinfer::log!("Testing device: {:?} (CPU reference)", device);
    let outputs = collect_output_names(&graph)?;
    let inputs = build_inputs(&model, &dynamic_vars)?;
    let grouped_outputs = group_outputs(&outputs);

    let sim_cpu = Simulator::new(&model, &graph, Device::Cpu)?;
    let mut exec_cpu = sim_cpu.make_executor()?;
    populate_exec(&mut exec_cpu, &inputs)?;
    exec_cpu.step()?;
    let refs = collect_outputs(&mut exec_cpu, &outputs)?;

    let sim = Simulator::new(&model, &graph, device)?;
    let mut exec = sim.make_executor()?;
    populate_exec(&mut exec, &inputs)?;

    exec.step()?;
    let trace = exec.trace();
    for (tag, names) in grouped_outputs {
        openinfer::log!("\n=== {} ===", tag);
        for name in names {
            let current: TensorValue = exec.fetch(&name)?;
            let reference = refs
                .get(&name)
                .ok_or_else(|| anyhow::anyhow!("missing ref {}", name))?;
            let status = compare_tensor(&current, reference);
            let current_fmt = format_truncated(&current);
            let ref_fmt = format_truncated(reference);
            openinfer::log!(
                "[{}] {}[0..10] = {} -- CPUref: {}",
                status,
                name,
                current_fmt,
                ref_fmt
            );
        }
    }
    openinfer::log!(
        "ops_v2 completed on {:?} with {} trace events",
        device,
        trace.len()
    );
    Ok(())
}

fn build_graph() -> Result<(Graph, Vec<DynamicVar>)> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);
    let mut dynamic_vars = Vec::new();

    let ops = [
        OpKind::Sub,
        OpKind::Div,
        OpKind::FloorDiv,
        OpKind::Rem,
        OpKind::Fma,
        OpKind::Neg,
        OpKind::Recip,
        OpKind::Sign,
        OpKind::Min,
        OpKind::Max,
        OpKind::Clamp,
        OpKind::Floor,
        OpKind::Ceil,
        OpKind::Round,
        OpKind::Trunc,
        OpKind::And,
        OpKind::Or,
        OpKind::Xor,
        OpKind::Not,
        OpKind::Shl,
        OpKind::Shr,
        OpKind::Popcount,
        OpKind::Eq,
        OpKind::Ne,
        OpKind::Lt,
        OpKind::Le,
        OpKind::Gt,
        OpKind::Ge,
        OpKind::Filter,
        OpKind::IsNan,
        OpKind::IsInf,
        OpKind::IsNeg,
        OpKind::SumAxis,
        OpKind::MeanAxis,
        OpKind::ProdAxis,
        OpKind::MaxAxis,
        OpKind::MinAxis,
        OpKind::ArgmaxAxis,
        OpKind::ArgminAxis,
    ];

    for op in ops {
        let schema = op_schema(op).ok_or_else(|| anyhow::anyhow!("missing schema for {:?}", op))?;
        let support = schema
            .dtype_support
            .ok_or_else(|| anyhow::anyhow!("missing dtype support for {:?}", op))?;
        let inputs = schema
            .inputs
            .fixed()
            .ok_or_else(|| anyhow::anyhow!("non-fixed input arity for {:?}", op))?;

        let allow_broadcast = schema.broadcast.allow() && inputs > 1;
        let allow_inplace = schema.inplace.allow() && !matches!(schema.type_rule, TypeRule::Fixed(_));
        let is_reduce = matches!(
            op,
            OpKind::SumAxis
                | OpKind::MeanAxis
                | OpKind::ProdAxis
                | OpKind::MaxAxis
                | OpKind::MinAxis
                | OpKind::ArgmaxAxis
                | OpKind::ArgminAxis
        );

        for &dtype in support.normal {
            let input_dims = if is_reduce {
                dims(&["R", "C"])
            } else {
                dims(&["V"])
            };
            let output_dims = if is_reduce {
                dims(&["R", "S"])
            } else {
                dims(&["V"])
            };

            let mut input_names = Vec::with_capacity(inputs);
            for idx in 0..inputs {
                let name = format!(
                    "{}_{}_in{}_norm",
                    op_tag(op),
                    dtype_tag(dtype),
                    idx
                );
                add_dynamic(
                    &mut graph,
                    &mut dynamic_vars,
                    &name,
                    dtype,
                    input_dims.clone(),
                );
                input_names.push(name);
            }

            let out_dtype = output_dtype(schema.type_rule, dtype);
            let out_name = format!("{}_{}_out_norm", op_tag(op), dtype_tag(dtype));
            add_volatile(&mut graph, &out_name, out_dtype, output_dims.clone());
            add_op_node(
                &mut graph,
                op,
                op_attrs(op, None),
                input_names.clone(),
                out_name,
            )?;

            if allow_broadcast {
                let mut bcast_inputs = Vec::with_capacity(inputs);
                let mut bcast_out_dims = dims(&["V"]);
                for idx in 0..inputs {
                    let dims = if op == OpKind::Div && dtype == DType::F8 {
                        bcast_out_dims = dims(&["R", "C"]);
                        if idx == 0 {
                            dims(&["R", "C"])
                        } else {
                            dims(&["C"])
                        }
                    } else if idx == 0 {
                        dims(&["V"])
                    } else {
                        dims(&["S"])
                    };
                    let name = format!(
                        "{}_{}_in{}_bcast",
                        op_tag(op),
                        dtype_tag(dtype),
                        idx
                    );
                    add_dynamic(&mut graph, &mut dynamic_vars, &name, dtype, dims);
                    bcast_inputs.push(name);
                }
                let out_name = format!("{}_{}_out_bcast", op_tag(op), dtype_tag(dtype));
                add_volatile(&mut graph, &out_name, out_dtype, bcast_out_dims);
                add_op_node(&mut graph, op, op_attrs(op, None), bcast_inputs, out_name)?;
            }

            if allow_inplace {
                let inplace_name =
                    format!("{}_{}_inplace", op_tag(op), dtype_tag(dtype));
                add_dynamic(
                    &mut graph,
                    &mut dynamic_vars,
                    &inplace_name,
                    dtype,
                    input_dims.clone(),
                );
                let mut inplace_inputs = vec![inplace_name.clone()];
                for idx in 1..inputs {
                    let name = format!(
                        "{}_{}_in{}_inplace",
                        op_tag(op),
                        dtype_tag(dtype),
                        idx
                    );
                    add_dynamic(
                        &mut graph,
                        &mut dynamic_vars,
                        &name,
                        dtype,
                        input_dims.clone(),
                    );
                    inplace_inputs.push(name);
                }
                add_op_node(
                    &mut graph,
                    op,
                    op_attrs(op, None),
                    inplace_inputs,
                    inplace_name,
                )?;
            }
        }

        if schema.accumulate.allow() {
            for &(in_dtype, acc_dtype) in support.accumulate {
                let input_dims = if is_reduce {
                    dims(&["R", "C"])
                } else {
                    dims(&["V"])
                };
                let output_dims = if is_reduce {
                    dims(&["R", "S"])
                } else {
                    dims(&["V"])
                };
                let mut acc_inputs = Vec::with_capacity(inputs);
                for idx in 0..inputs {
                    let name = format!(
                        "{}_{}_in{}_acc",
                        op_tag(op),
                        dtype_tag(in_dtype),
                        idx
                    );
                    add_dynamic(
                        &mut graph,
                        &mut dynamic_vars,
                        &name,
                        in_dtype,
                        input_dims.clone(),
                    );
                    acc_inputs.push(name);
                }
                let out_name = format!(
                    "{}_{}_out_acc_{}",
                    op_tag(op),
                    dtype_tag(in_dtype),
                    dtype_tag(acc_dtype)
                );
                add_volatile(&mut graph, &out_name, acc_dtype, output_dims.clone());
                add_op_node(
                    &mut graph,
                    op,
                    op_attrs(op, Some(acc_dtype)),
                    acc_inputs.clone(),
                    out_name,
                )?;

                if allow_broadcast {
                    let mut acc_bcast_inputs = Vec::with_capacity(inputs);
                    for idx in 0..inputs {
                        let dims = if idx == 0 {
                            dims(&["V"])
                        } else {
                            dims(&["S"])
                        };
                        let name = format!(
                            "{}_{}_in{}_acc_bcast",
                            op_tag(op),
                            dtype_tag(in_dtype),
                            idx
                        );
                        add_dynamic(&mut graph, &mut dynamic_vars, &name, in_dtype, dims);
                        acc_bcast_inputs.push(name);
                    }
                    let out_name = format!(
                        "{}_{}_out_acc_bcast_{}",
                        op_tag(op),
                        dtype_tag(in_dtype),
                        dtype_tag(acc_dtype)
                    );
                    add_volatile(&mut graph, &out_name, acc_dtype, dims(&["V"]));
                    add_op_node(
                        &mut graph,
                        op,
                        op_attrs(op, Some(acc_dtype)),
                        acc_bcast_inputs,
                        out_name,
                    )?;
                }
            }
        }
    }

    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;
    Ok((graph, dynamic_vars))
}

fn add_dynamic(
    graph: &mut Graph,
    vars: &mut Vec<DynamicVar>,
    name: &str,
    dtype: DType,
    dims: Vec<String>,
) {
    graph.add_var(
        MemoryKind::Dynamic,
        name,
        dtype,
        dims.clone(),
        None,
        None,
        Vec::new(),
        None,
        false,
        Vec::new(),
        Vec::new(),
    );
    vars.push(DynamicVar {
        name: name.to_string(),
        dtype,
        dims,
    });
}

fn add_volatile(graph: &mut Graph, name: &str, dtype: DType, dims: Vec<String>) {
    graph.add_var(
        MemoryKind::Volatile,
        name,
        dtype,
        dims,
        None,
        None,
        Vec::new(),
        None,
        false,
        Vec::new(),
        Vec::new(),
    );
}

fn add_op_node(
    graph: &mut Graph,
    op: OpKind,
    attrs: OpAttrs,
    inputs: Vec<String>,
    output: String,
) -> Result<()> {
    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op,
            attrs,
            inputs,
            output,
        },
    )
}

fn op_attrs(op: OpKind, acc: Option<DType>) -> OpAttrs {
    let mut items = Vec::new();
    match op {
        OpKind::Div | OpKind::FloorDiv | OpKind::Recip => {
            items.push(OpAttr {
                name: "div_by_zero_mask".to_string(),
                value: AttrValue::Int(0),
            });
        }
        OpKind::Clamp => {
            items.push(OpAttr {
                name: "min".to_string(),
                value: AttrValue::Int(0),
            });
            items.push(OpAttr {
                name: "max".to_string(),
                value: AttrValue::UInt(6),
            });
        }
        OpKind::Shl | OpKind::Shr => {
            items.push(OpAttr {
                name: "bits".to_string(),
                value: AttrValue::Int(1),
            });
        }
        OpKind::SumAxis
        | OpKind::MeanAxis
        | OpKind::ProdAxis
        | OpKind::MaxAxis
        | OpKind::MinAxis => {
            items.push(OpAttr {
                name: "axes".to_string(),
                value: AttrValue::IntList(vec![1]),
            });
            items.push(OpAttr {
                name: "keepdims".to_string(),
                value: AttrValue::Bool(true),
            });
        }
        OpKind::ArgmaxAxis | OpKind::ArgminAxis => {
            items.push(OpAttr {
                name: "axis".to_string(),
                value: AttrValue::Int(1),
            });
            items.push(OpAttr {
                name: "keepdims".to_string(),
                value: AttrValue::Bool(true),
            });
            items.push(OpAttr {
                name: "select_first".to_string(),
                value: AttrValue::Bool(true),
            });
        }
        _ => {}
    }
    if let Some(dtype) = acc {
        items.push(OpAttr {
            name: "acc".to_string(),
            value: AttrValue::DType(dtype),
        });
    }
    OpAttrs { items }
}

fn output_dtype(rule: TypeRule, input: DType) -> DType {
    match rule {
        TypeRule::SameAsInput(_) => input,
        TypeRule::Fixed(dtype) => dtype,
        TypeRule::AccFromAttr { .. } => input,
    }
}

fn resolve_dims(model: &ModelLoader, dims: &[String]) -> Result<Vec<usize>> {
    let mut shape = Vec::with_capacity(dims.len());
    for dim in dims {
        if let Ok(value) = dim.parse::<usize>() {
            shape.push(value);
        } else {
            shape.push(model.size_of(dim)?);
        }
    }
    Ok(shape)
}

#[derive(Clone, Copy)]
struct FloatTol {
    abs: f64,
    rel: f64,
}

impl FloatTol {
    fn f16() -> Self {
        Self { abs: 0.6, rel: 0.08 }
    }

    fn bf16() -> Self {
        Self { abs: 0.1, rel: 0.02 }
    }

    fn f8() -> Self {
        Self { abs: 0.6, rel: 0.25 }
    }

    fn f32() -> Self {
        Self { abs: 1e-4, rel: 1e-4 }
    }

    fn f64() -> Self {
        Self { abs: 1e-8, rel: 1e-8 }
    }
}

fn within_tol(a: f64, b: f64, tol: FloatTol) -> bool {
    let diff = (a - b).abs();
    if diff <= tol.abs {
        return true;
    }
    let scale = a.abs().max(b.abs());
    diff <= tol.rel * scale
}

fn float_status(current: &[f64], reference: &[f64], tol: FloatTol) -> &'static str {
    if current.len() != reference.len() {
        return "❌";
    }
    let mut exact = true;
    for (a, b) in current.iter().zip(reference.iter()) {
        if a.is_nan() && b.is_nan() {
            continue;
        }
        if a == b {
            continue;
        }
        exact = false;
        if !within_tol(*a, *b, tol) {
            return "❌";
        }
    }
    if exact { "✅" } else { "⚠️" }
}

fn collect_output_names(graph: &Graph) -> Result<Vec<String>> {
    let block = graph.block(ENTRY_BLOCK)?;
    let mut outputs = BTreeSet::new();
    for node in &block.nodes {
        if let NodeKind::Op { output, .. } = &node.kind {
            outputs.insert(output.clone());
        }
    }
    Ok(outputs.into_iter().collect())
}

fn build_inputs(model: &ModelLoader, vars: &[DynamicVar]) -> Result<Vec<(String, TensorValue)>> {
    let mut inputs = Vec::with_capacity(vars.len());
    for var in vars {
        let shape = resolve_dims(model, &var.dims)?;
        let seed = seed_from_name(&var.name);
        let tensor = make_tensor(var.dtype, &shape, seed);
        inputs.push((var.name.clone(), tensor));
    }
    Ok(inputs)
}

fn seed_from_name(name: &str) -> u8 {
    name.bytes().fold(0u8, |acc, b| acc.wrapping_add(b))
}

fn make_tensor(dtype: DType, shape: &[usize], seed: u8) -> TensorValue {
    let mut value = TensorValue::zeros(dtype, shape);
    fill_tensor(&mut value, seed);
    value
}

fn fill_tensor(value: &mut TensorValue, seed: u8) {
    match value {
        TensorValue::I8(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = (seed as i8).wrapping_add(i as i8).wrapping_sub(4);
            }
        }
        TensorValue::I16(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = (seed as i16).wrapping_add(i as i16).wrapping_sub(4);
            }
        }
        TensorValue::I32(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = (seed as i32).wrapping_add(i as i32).wrapping_sub(4);
            }
        }
        TensorValue::I64(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = (seed as i64).wrapping_add(i as i64).wrapping_sub(4);
            }
        }
        TensorValue::U8(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = seed.wrapping_add(i as u8);
            }
        }
        TensorValue::U16(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = (seed as u16).wrapping_add(i as u16);
            }
        }
        TensorValue::U32(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = (seed as u32).wrapping_add(i as u32);
            }
        }
        TensorValue::U64(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = (seed as u64).wrapping_add(i as u64);
            }
        }
        TensorValue::Bool(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                let toggle = (i as u8).wrapping_add(seed) % 3 == 0;
                *v = toggle;
            }
        }
        TensorValue::Bitset(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                v.bits = seed.wrapping_add(i as u8).wrapping_mul(3).wrapping_add(1);
            }
        }
        TensorValue::F16(tensor) => {
            let base = seed as f32 * 0.1 - 1.0;
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = F16::from_f32(base + i as f32 * 0.25);
            }
        }
        TensorValue::BF16(tensor) => {
            let base = seed as f32 * 0.1 - 1.0;
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = BF16::from_f32(base + i as f32 * 0.25);
            }
        }
        TensorValue::F8(tensor) => {
            let base = seed as f32 * 0.1 - 1.0;
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = F8::from_f32(base + i as f32 * 0.25);
            }
        }
        TensorValue::F32(tensor) => {
            let base = seed as f32 * 0.1 - 1.0;
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = base + i as f32 * 0.25;
            }
        }
        TensorValue::F64(tensor) => {
            let base = seed as f64 * 0.1 - 1.0;
            for (i, v) in tensor.data.iter_mut().enumerate() {
                *v = base + i as f64 * 0.25;
            }
        }
        TensorValue::I4(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                v.bits = seed.wrapping_add(i as u8).wrapping_add(1);
            }
        }
        TensorValue::I2(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                v.bits = seed.wrapping_add(i as u8).wrapping_add(1);
            }
        }
        TensorValue::I1(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                v.bits = seed.wrapping_add(i as u8).wrapping_add(1);
            }
        }
        TensorValue::U4(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                v.bits = seed.wrapping_add(i as u8).wrapping_add(1);
            }
        }
        TensorValue::U2(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                v.bits = seed.wrapping_add(i as u8).wrapping_add(1);
            }
        }
        TensorValue::U1(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                v.bits = seed.wrapping_add(i as u8).wrapping_add(1);
            }
        }
        TensorValue::T1(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                v.bits = seed.wrapping_add(i as u8).wrapping_add(1);
            }
        }
        TensorValue::T2(tensor) => {
            for (i, v) in tensor.data.iter_mut().enumerate() {
                v.bits = seed.wrapping_add(i as u8).wrapping_add(1);
            }
        }
    }
}

fn populate_exec(exec: &mut Executor, inputs: &[(String, TensorValue)]) -> Result<()> {
    for (name, tensor) in inputs {
        exec.insert_dynamic(name, tensor.clone())?;
    }
    Ok(())
}

fn collect_outputs(exec: &mut Executor, names: &[String]) -> Result<HashMap<String, TensorValue>> {
    let mut refs = HashMap::with_capacity(names.len());
    for name in names {
        let tensor: TensorValue = exec.fetch(name)?;
        refs.insert(name.clone(), tensor);
    }
    Ok(refs)
}

fn float_values(value: &TensorValue) -> Option<Vec<f64>> {
    match value {
        TensorValue::F16(tensor) => Some(tensor.data.iter().map(|v| v.to_f32() as f64).collect()),
        TensorValue::BF16(tensor) => Some(tensor.data.iter().map(|v| v.to_f32() as f64).collect()),
        TensorValue::F8(tensor) => Some(tensor.data.iter().map(|v| v.to_f32() as f64).collect()),
        TensorValue::F32(tensor) => Some(tensor.data.iter().map(|v| *v as f64).collect()),
        TensorValue::F64(tensor) => Some(tensor.data.clone()),
        _ => None,
    }
}

fn format_truncated(value: &TensorValue) -> String {
    match value {
        TensorValue::I8(tensor) => format_truncated_slice(&tensor.data, |v| v.to_string()),
        TensorValue::I16(tensor) => format_truncated_slice(&tensor.data, |v| v.to_string()),
        TensorValue::I32(tensor) => format_truncated_slice(&tensor.data, |v| v.to_string()),
        TensorValue::I64(tensor) => format_truncated_slice(&tensor.data, |v| v.to_string()),
        TensorValue::U8(tensor) => format_truncated_slice(&tensor.data, |v| v.to_string()),
        TensorValue::U16(tensor) => format_truncated_slice(&tensor.data, |v| v.to_string()),
        TensorValue::U32(tensor) => format_truncated_slice(&tensor.data, |v| v.to_string()),
        TensorValue::U64(tensor) => format_truncated_slice(&tensor.data, |v| v.to_string()),
        TensorValue::Bool(tensor) => format_truncated_slice(&tensor.data, |v| v.to_string()),
        TensorValue::Bitset(tensor) => {
            format_truncated_slice(&tensor.data, |v| v.bits.to_string())
        }
        TensorValue::F16(tensor) => {
            format_truncated_slice(&tensor.data, |v| v.to_f32().to_string())
        }
        TensorValue::BF16(tensor) => {
            format_truncated_slice(&tensor.data, |v| v.to_f32().to_string())
        }
        TensorValue::F8(tensor) => {
            format_truncated_slice(&tensor.data, |v| v.to_f32().to_string())
        }
        TensorValue::F32(tensor) => format_truncated_slice(&tensor.data, |v| v.to_string()),
        TensorValue::F64(tensor) => format_truncated_slice(&tensor.data, |v| v.to_string()),
        TensorValue::I4(tensor) => format_truncated_slice(&tensor.data, |v| v.bits.to_string()),
        TensorValue::I2(tensor) => format_truncated_slice(&tensor.data, |v| v.bits.to_string()),
        TensorValue::I1(tensor) => format_truncated_slice(&tensor.data, |v| v.bits.to_string()),
        TensorValue::U4(tensor) => format_truncated_slice(&tensor.data, |v| v.bits.to_string()),
        TensorValue::U2(tensor) => format_truncated_slice(&tensor.data, |v| v.bits.to_string()),
        TensorValue::U1(tensor) => format_truncated_slice(&tensor.data, |v| v.bits.to_string()),
        TensorValue::T1(tensor) => format_truncated_slice(&tensor.data, |v| v.bits.to_string()),
        TensorValue::T2(tensor) => format_truncated_slice(&tensor.data, |v| v.bits.to_string()),
    }
}

fn format_truncated_slice<T, F>(data: &[T], fmt: F) -> String
where
    F: Fn(&T) -> String,
{
    let len = data.len();
    let mut items = Vec::new();
    if len <= 10 {
        for item in data {
            items.push(fmt(item));
        }
    } else {
        for item in &data[..5] {
            items.push(fmt(item));
        }
        items.push("...".to_string());
        for item in &data[len - 5..] {
            items.push(fmt(item));
        }
    }
    format!("[{}]", items.join(", "))
}

fn compare_tensor(current: &TensorValue, reference: &TensorValue) -> &'static str {
    if current.dtype() != reference.dtype() || current.shape() != reference.shape() {
        return "❌";
    }
    match (current, reference) {
        (TensorValue::F16(_), _) => match (float_values(current), float_values(reference)) {
            (Some(current_vals), Some(reference_vals)) => {
                float_status(&current_vals, &reference_vals, FloatTol::f16())
            }
            _ => "❌",
        },
        (TensorValue::BF16(_), _) => match (float_values(current), float_values(reference)) {
            (Some(current_vals), Some(reference_vals)) => {
                float_status(&current_vals, &reference_vals, FloatTol::bf16())
            }
            _ => "❌",
        },
        (TensorValue::F8(_), _) => match (float_values(current), float_values(reference)) {
            (Some(current_vals), Some(reference_vals)) => {
                float_status(&current_vals, &reference_vals, FloatTol::f8())
            }
            _ => "❌",
        },
        (TensorValue::F32(_), _) => match (float_values(current), float_values(reference)) {
            (Some(current_vals), Some(reference_vals)) => {
                float_status(&current_vals, &reference_vals, FloatTol::f32())
            }
            _ => "❌",
        },
        (TensorValue::F64(_), _) => match (float_values(current), float_values(reference)) {
            (Some(current_vals), Some(reference_vals)) => {
                float_status(&current_vals, &reference_vals, FloatTol::f64())
            }
            _ => "❌",
        },
        (TensorValue::I8(a), TensorValue::I8(b)) => exact_status(a.data == b.data),
        (TensorValue::I16(a), TensorValue::I16(b)) => exact_status(a.data == b.data),
        (TensorValue::I32(a), TensorValue::I32(b)) => exact_status(a.data == b.data),
        (TensorValue::I64(a), TensorValue::I64(b)) => exact_status(a.data == b.data),
        (TensorValue::U8(a), TensorValue::U8(b)) => exact_status(a.data == b.data),
        (TensorValue::U16(a), TensorValue::U16(b)) => exact_status(a.data == b.data),
        (TensorValue::U32(a), TensorValue::U32(b)) => exact_status(a.data == b.data),
        (TensorValue::U64(a), TensorValue::U64(b)) => exact_status(a.data == b.data),
        (TensorValue::Bool(a), TensorValue::Bool(b)) => exact_status(a.data == b.data),
        (TensorValue::Bitset(a), TensorValue::Bitset(b)) => exact_status(a.data == b.data),
        (TensorValue::I4(a), TensorValue::I4(b)) => exact_status(a.data == b.data),
        (TensorValue::I2(a), TensorValue::I2(b)) => exact_status(a.data == b.data),
        (TensorValue::I1(a), TensorValue::I1(b)) => exact_status(a.data == b.data),
        (TensorValue::U4(a), TensorValue::U4(b)) => exact_status(a.data == b.data),
        (TensorValue::U2(a), TensorValue::U2(b)) => exact_status(a.data == b.data),
        (TensorValue::U1(a), TensorValue::U1(b)) => exact_status(a.data == b.data),
        (TensorValue::T1(a), TensorValue::T1(b)) => exact_status(a.data == b.data),
        (TensorValue::T2(a), TensorValue::T2(b)) => exact_status(a.data == b.data),
        _ => "❌",
    }
}

fn exact_status(equal: bool) -> &'static str {
    if equal { "✅" } else { "❌" }
}

fn group_outputs(outputs: &[String]) -> BTreeMap<String, Vec<String>> {
    let mut grouped: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for name in outputs {
        let tag = op_group(name);
        grouped.entry(tag.to_string()).or_default().push(name.clone());
    }
    grouped
}

fn op_group(name: &str) -> &'static str {
    for tag in op_tags_sorted() {
        if name.starts_with(tag) && name.as_bytes().get(tag.len()) == Some(&b'_') {
            return tag;
        }
    }
    "unknown"
}

fn op_tags_sorted() -> &'static [&'static str] {
    &[
        "floor_div",
        "argmax_axis",
        "argmin_axis",
        "mean_axis",
        "prod_axis",
        "max_axis",
        "min_axis",
        "sum_axis",
        "is_nan",
        "is_inf",
        "is_neg",
        "popcount",
        "clamp",
        "ceil",
        "round",
        "trunc",
        "filter",
        "recip",
        "sign",
        "floor",
        "sub",
        "div",
        "rem",
        "fma",
        "neg",
        "min",
        "max",
        "and",
        "or",
        "xor",
        "not",
        "shl",
        "shr",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
    ]
}

fn dims(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

fn op_tag(op: OpKind) -> &'static str {
    match op {
        OpKind::Sub => "sub",
        OpKind::Div => "div",
        OpKind::FloorDiv => "floor_div",
        OpKind::Rem => "rem",
        OpKind::Fma => "fma",
        OpKind::Neg => "neg",
        OpKind::Recip => "recip",
        OpKind::Sign => "sign",
        OpKind::Min => "min",
        OpKind::Max => "max",
        OpKind::Clamp => "clamp",
        OpKind::Floor => "floor",
        OpKind::Ceil => "ceil",
        OpKind::Round => "round",
        OpKind::Trunc => "trunc",
        OpKind::And => "and",
        OpKind::Or => "or",
        OpKind::Xor => "xor",
        OpKind::Not => "not",
        OpKind::Shl => "shl",
        OpKind::Shr => "shr",
        OpKind::Popcount => "popcount",
        OpKind::Eq => "eq",
        OpKind::Ne => "ne",
        OpKind::Lt => "lt",
        OpKind::Le => "le",
        OpKind::Gt => "gt",
        OpKind::Ge => "ge",
        OpKind::Filter => "filter",
        OpKind::IsNan => "is_nan",
        OpKind::IsInf => "is_inf",
        OpKind::IsNeg => "is_neg",
        OpKind::SumAxis => "sum_axis",
        OpKind::MeanAxis => "mean_axis",
        OpKind::ProdAxis => "prod_axis",
        OpKind::MaxAxis => "max_axis",
        OpKind::MinAxis => "min_axis",
        OpKind::ArgmaxAxis => "argmax_axis",
        OpKind::ArgminAxis => "argmin_axis",
        _ => "unknown",
    }
}

fn dtype_tag(dtype: DType) -> &'static str {
    match dtype {
        DType::I8 => "i8",
        DType::I16 => "i16",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::U8 => "u8",
        DType::U16 => "u16",
        DType::U32 => "u32",
        DType::U64 => "u64",
        DType::F8 => "f8",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::Bool => "bool",
        DType::I1 => "i1",
        DType::I2 => "i2",
        DType::I4 => "i4",
        DType::U1 => "u1",
        DType::U2 => "u2",
        DType::U4 => "u4",
        DType::T1 => "t1",
        DType::T2 => "t2",
        DType::Bitset => "bitset",
    }
}
