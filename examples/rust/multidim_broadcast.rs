use openinfer::{
    fetch_executor, graph, insert_executor, FormatValue, ModelLoader, Simulator, Tensor,
    TensorOptions,
};
use std::path::Path;

mod util;
use util::select_device;

fn format_head_tail<T: FormatValue>(data: &[T], head: usize, tail: usize) -> String {
    let len = data.len();
    if len == 0 {
        return "{}".to_string();
    }
    if len <= head + tail {
        let joined = data
            .iter()
            .map(FormatValue::format_value)
            .collect::<Vec<_>>()
            .join(", ");
        return format!("{{{}}}", joined);
    }
    let head_vals = data[..head]
        .iter()
        .map(FormatValue::format_value)
        .collect::<Vec<_>>()
        .join(", ");
    let tail_vals = data[len - tail..]
        .iter()
        .map(FormatValue::format_value)
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{{} ... {}}}", head_vals, tail_vals)
}

fn print_head_tail<T: FormatValue>(label: &str, data: &[T]) {
    log::info!("{label} = {}", format_head_tail(data, 5, 5));
}

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/multidim_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[A, B];
            y: f32[B];
            z: f32[A, 1];
        }

        volatile {
            out: f32[A, B];
        }

        block entry {
            op add(x, y) >> out;
            op mul(out, z) >> out;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let a = model.size_of("A")?;
    let b = model.size_of("B")?;
    let x_shape = vec![a, b];
    let y_shape = vec![b];
    let z_shape = vec![a, 1];

    let x_data: Vec<f32> = (0..(a * b)).map(|i| i as f32 * 0.1).collect();
    let y_data: Vec<f32> = (0..b).map(|i| i as f32 * 0.01).collect();
    let z_data: Vec<f32> = (0..a).map(|i| 1.0 + i as f32).collect();

    let x = Tensor::from_vec_with_opts(
        x_data,
        TensorOptions {
            shape: Some(x_shape.clone()),
            ..TensorOptions::default()
        },
    )?;
    let y = Tensor::from_vec_with_opts(
        y_data,
        TensorOptions {
            shape: Some(y_shape),
            ..TensorOptions::default()
        },
    )?;
    let z = Tensor::from_vec_with_opts(
        z_data,
        TensorOptions {
            shape: Some(z_shape),
            ..TensorOptions::default()
        },
    )?;

    print_head_tail("x", &x.to_vec());
    print_head_tail("y", &y.to_vec());
    print_head_tail("z", &z.to_vec());

    insert_executor!(exec, { x: x, y: y, z: z });
    exec.step()?;

    fetch_executor!(exec, { out: Tensor<f32> });
    log::info!("out shape = {:?}", out.shape());
    print_head_tail("out", &out.to_vec());

    Ok(())
}
