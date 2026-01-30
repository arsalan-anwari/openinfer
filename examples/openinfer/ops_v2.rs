use anyhow::Result;
use openinfer::{
    op_schema, AttrValue, DType, Graph, MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs, OpKind,
    Simulator, TensorValue, TypeRule,
};
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
    let sim = Simulator::new(&model, &graph, device)?.with_trace();
    let mut exec = sim.make_executor()?;

    for var in dynamic_vars {
        let shape = resolve_dims(&model, &var.dims)?;
        let tensor = TensorValue::zeros(var.dtype, &shape);
        exec.insert_dynamic(&var.name, tensor)?;
    }

    exec.step()?;
    let trace = exec.trace();
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
                for idx in 0..inputs {
                    let dims = if idx == 0 {
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
                add_volatile(&mut graph, &out_name, out_dtype, dims(&["V"]));
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
