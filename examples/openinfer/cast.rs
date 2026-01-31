use anyhow::Result;
use openinfer::{
    AttrValue, DType, Graph, MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs, OpKind, Simulator,
    Tensor, TensorOptions, TensorValue, I4, U2,
};
use std::path::Path;

mod util;
use util::select_device;

const ENTRY_BLOCK: &str = "entry";

fn main() -> Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/cast_model.oinf");
    let model = ModelLoader::open(model_path)?;
    let graph = build_graph()?;

    let device = select_device()?;
    let sim = Simulator::new(&model, &graph, device)?;
    let mut exec = sim.make_executor()?;

    let len = model.size_of("N")?;
    let shape = vec![len];

    let packed_i4 = pack_i4(&resize_i8_values(&[-8, -3, -1, 0, 1, 3, 7, 2], len), &shape)?;
    let packed_u2 = pack_u2(&resize_u8_values(&[0, 1, 2, 3, 2, 1, 0, 3], len), &shape)?;

    let i4_source = resize_i8_values(&[-8, -3, -1, 0, 1, 3, 7, 2], len);
    let u2_source = resize_u8_values(&[0, 1, 2, 3, 2, 1, 0, 3], len);
    let i8_input = Tensor::from_vec_with_opts(
        resize_i8_values(&[-5, 3, 7, 100, -120, 0, 1, 2], len),
        TensorOptions {
            shape: Some(shape.clone()),
            ..TensorOptions::default()
        },
    )?;
    let f32_input = Tensor::from_vec_with_opts(
        resize_f32_values(&[-5.5, 0.0, 3.2, 260.0, 255.0, 128.1, 1.0, 1000.0], len),
        TensorOptions {
            shape: Some(shape.clone()),
            ..TensorOptions::default()
        },
    )?;

    exec.insert_dynamic("in_i4", TensorValue::I4(packed_i4))?;
    exec.insert_dynamic("in_u2", TensorValue::U2(packed_u2))?;
    let i8_input_data = i8_input.data.clone();
    let f32_input_data = f32_input.data.clone();
    exec.insert_dynamic("in_i8", TensorValue::I8(i8_input))?;
    exec.insert_dynamic("in_f32", TensorValue::F32(f32_input))?;

    exec.step()?;

    let out_i32 = exec.fetch_typed::<i32>("out_i32")?;
    let out_f32 = exec.fetch_typed::<f32>("out_f32")?;
    let out_f64 = exec.fetch_typed::<f64>("out_f64")?;
    let out_u8 = exec.fetch_typed::<u8>("out_u8")?;
    let out_i64 = exec.fetch_typed::<i64>("out_i64")?;

    openinfer::log!("cast i4 -> i32 input={:?} output={:?}", i4_source, out_i32.data);
    openinfer::log!("cast u2 -> f32 input={:?} output={:?}", u2_source, out_f32.data);
    openinfer::log!("cast i8 -> f64 input={:?} output={:?}", i8_input_data, out_f64.data);
    openinfer::log!(
        "cast f32 -> u8 input={:?} output={:?}",
        f32_input_data,
        out_u8.data
    );
    openinfer::log!(
        "cast f32 -> i64 input={:?} output={:?}",
        f32_input_data,
        out_i64.data
    );

    match run_invalid_cast(&model, device, &shape) {
        Ok(()) => openinfer::log!("unexpected success: invalid cast should fail"),
        Err(err) => openinfer::log!("expected error: {}", err),
    }

    Ok(())
}

fn build_graph() -> Result<Graph> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    add_dynamic(&mut graph, "in_i4", DType::I4, dims(&["N"]));
    add_dynamic(&mut graph, "in_u2", DType::U2, dims(&["N"]));
    add_dynamic(&mut graph, "in_i8", DType::I8, dims(&["N"]));
    add_dynamic(&mut graph, "in_f32", DType::F32, dims(&["N"]));

    add_volatile(&mut graph, "out_i32", DType::I32, dims(&["N"]));
    add_volatile(&mut graph, "out_f32", DType::F32, dims(&["N"]));
    add_volatile(&mut graph, "out_f64", DType::F64, dims(&["N"]));
    add_volatile(&mut graph, "out_u8", DType::U8, dims(&["N"]));
    add_volatile(&mut graph, "out_i64", DType::I64, dims(&["N"]));

    add_op_node(
        &mut graph,
        OpKind::Cast,
        cast_attrs(DType::I32),
        vec!["in_i4".to_string()],
        "out_i32".to_string(),
    )?;
    add_op_node(
        &mut graph,
        OpKind::Cast,
        cast_attrs(DType::F32),
        vec!["in_u2".to_string()],
        "out_f32".to_string(),
    )?;
    add_op_node(
        &mut graph,
        OpKind::Cast,
        cast_attrs(DType::F64),
        vec!["in_i8".to_string()],
        "out_f64".to_string(),
    )?;
    add_op_node(
        &mut graph,
        OpKind::Cast,
        cast_attrs(DType::U8),
        vec!["in_f32".to_string()],
        "out_u8".to_string(),
    )?;
    add_op_node(
        &mut graph,
        OpKind::Cast,
        cast_attrs(DType::I64),
        vec!["in_f32".to_string()],
        "out_i64".to_string(),
    )?;

    Ok(graph)
}

fn run_invalid_cast(model: &ModelLoader, device: openinfer::Device, shape: &[usize]) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);
    add_dynamic(&mut graph, "bad_in", DType::I8, dims(&["N"]));
    add_volatile(&mut graph, "bad_out", DType::U8, dims(&["N"]));
    add_op_node(
        &mut graph,
        OpKind::Cast,
        cast_attrs(DType::U8),
        vec!["bad_in".to_string()],
        "bad_out".to_string(),
    )?;

    let sim = Simulator::new(model, &graph, device)?;
    let mut exec = sim.make_executor()?;
    let input = Tensor::from_vec_with_opts(
        resize_i8_values(&[1, 2, 3, 4, 5, 6, 7, 8], shape[0]),
        TensorOptions {
            shape: Some(shape.to_vec()),
            ..TensorOptions::default()
        },
    )?;
    exec.insert_dynamic("bad_in", TensorValue::I8(input))?;
    exec.step().map(|_| ())
}

fn cast_attrs(to: DType) -> OpAttrs {
    OpAttrs {
        items: vec![OpAttr {
            name: "to".to_string(),
            value: AttrValue::DType(to),
        }],
    }
}

fn add_dynamic(graph: &mut Graph, name: &str, dtype: DType, dims: Vec<String>) {
    graph.add_var(
        MemoryKind::Dynamic,
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

fn dims(names: &[&str]) -> Vec<String> {
    names.iter().map(|name| (*name).to_string()).collect()
}

fn resize_i8_values(values: &[i8], len: usize) -> Vec<i8> {
    resize_values(values, len, 0i8)
}

fn resize_u8_values(values: &[u8], len: usize) -> Vec<u8> {
    resize_values(values, len, 0u8)
}

fn resize_f32_values(values: &[f32], len: usize) -> Vec<f32> {
    resize_values(values, len, 0.0f32)
}

fn resize_values<T: Copy>(values: &[T], len: usize, fill: T) -> Vec<T> {
    if values.len() == len {
        return values.to_vec();
    }
    if values.len() > len {
        return values[..len].to_vec();
    }
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(values);
    out.resize(len, fill);
    out
}

fn pack_i4(values: &[i8], shape: &[usize]) -> Result<Tensor<I4>> {
    let logical_len = numel(shape);
    let packed_len = DType::I4.storage_len(logical_len);
    let mut data = vec![I4 { bits: 0 }; packed_len];
    for idx in 0..logical_len {
        let value = values[idx];
        let raw = I4::from_i8(value).bits;
        set_bits(&mut data, idx, 4, raw);
    }
    Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(shape.to_vec()),
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )
}

fn pack_u2(values: &[u8], shape: &[usize]) -> Result<Tensor<U2>> {
    let logical_len = numel(shape);
    let packed_len = DType::U2.storage_len(logical_len);
    let mut data = vec![U2 { bits: 0 }; packed_len];
    for idx in 0..logical_len {
        let value = values[idx];
        let raw = U2::from_u8(value).bits;
        set_bits(&mut data, idx, 2, raw);
    }
    Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(shape.to_vec()),
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )
}

fn numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        return 1;
    }
    shape.iter().product()
}

fn set_bits<T: PackedBits>(data: &mut [T], index: usize, width: u8, value: u8) {
    let per_byte = 8 / width;
    let byte_index = index / per_byte as usize;
    let bit_index = (index % per_byte as usize) as u8;
    let shift = bit_index * width;
    let mask = (1u8 << width) - 1;
    let mut byte = data[byte_index].bits();
    byte &= !(mask << shift);
    byte |= (value & mask) << shift;
    data[byte_index].set_bits(byte);
}

trait PackedBits: Copy {
    fn bits(&self) -> u8;
    fn set_bits(&mut self, value: u8);
}

impl PackedBits for I4 {
    fn bits(&self) -> u8 {
        self.bits
    }

    fn set_bits(&mut self, value: u8) {
        self.bits = value;
    }
}

impl PackedBits for U2 {
    fn bits(&self) -> u8 {
        self.bits
    }

    fn set_bits(&mut self, value: u8) {
        self.bits = value;
    }
}
