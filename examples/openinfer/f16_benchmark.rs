use openinfer::{graph, insert_executor, Device, F16, ModelLoader, Random, Simulator, Tensor};
use std::path::Path;

mod util;
use util::select_device;

struct Inputs {
    x0: Tensor<F16>,
    y0: Tensor<F16>,
    x1: Tensor<F16>,
    y1: Tensor<F16>,
    x2: Tensor<F16>,
    y2: Tensor<F16>,
    x3: Tensor<F16>,
    y3: Tensor<F16>,
}

fn build_inputs(model: &ModelLoader) -> anyhow::Result<Inputs> {
    let n0 = model.size_of("N0")?;
    let n1 = model.size_of("N1")?;
    let n2 = model.size_of("N2")?;
    let n3 = model.size_of("N3")?;

    let x0 = Random::<F16>::generate_with_seed(0, (F16::from_f32(-1.0), F16::from_f32(1.0)), n0)?;
    let y0 = Random::<F16>::generate_with_seed(1, (F16::from_f32(-1.0), F16::from_f32(1.0)), n0)?;
    let x1 = Random::<F16>::generate_with_seed(2, (F16::from_f32(-1.0), F16::from_f32(1.0)), n1)?;
    let y1 = Random::<F16>::generate_with_seed(3, (F16::from_f32(-1.0), F16::from_f32(1.0)), n1)?;
    let x2 = Random::<F16>::generate_with_seed(4, (F16::from_f32(-1.0), F16::from_f32(1.0)), n2)?;
    let y2 = Random::<F16>::generate_with_seed(5, (F16::from_f32(-1.0), F16::from_f32(1.0)), n2)?;
    let x3 = Random::<F16>::generate_with_seed(6, (F16::from_f32(-1.0), F16::from_f32(1.0)), n3)?;
    let y3 = Random::<F16>::generate_with_seed(7, (F16::from_f32(-1.0), F16::from_f32(1.0)), n3)?;

    Ok(Inputs {
        x0,
        y0,
        x1,
        y1,
        x2,
        y2,
        x3,
        y3,
    })
}

fn run_benchmark(
    label: &str,
    model: &ModelLoader,
    graph: &openinfer::Graph,
    device: Device,
    inputs: &Inputs,
    simulated: bool,
) -> anyhow::Result<()> {
    openinfer::trace!("\n=== {} ===", label);
    let sim = Simulator::new(model, graph, device)?;
    let sim = if simulated {
        sim.with_trace().with_timer().with_simulated_float()
    } else {
        sim.with_trace().with_timer()
    };
    let mut exec = sim.make_executor()?;

    insert_executor!(exec, {
        x0: inputs.x0.clone(),
        y0: inputs.y0.clone(),
        x1: inputs.x1.clone(),
        y1: inputs.y1.clone(),
        x2: inputs.x2.clone(),
        y2: inputs.y2.clone(),
        x3: inputs.x3.clone(),
        y3: inputs.y3.clone(),
    });
    exec.step()?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/f16_benchmark_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x0: f16[N0];
            y0: f16[N0];
            x1: f16[N1];
            y1: f16[N1];
            x2: f16[N2];
            y2: f16[N2];
            x3: f16[N3];
            y3: f16[N3];
        }

        volatile {
            z0: f16[N0];
            z1: f16[N1];
            z2: f16[N2];
            z3: f16[N3];
        }

        block entry {
            op add(x0, y0) >> z0;
            op add(x1, y1) >> z1;
            op add(x2, y2) >> z2;
            op add(x3, y3) >> z3;
            return;
        }
    };

    let device = select_device()?;
    let inputs = build_inputs(&model)?;

    run_benchmark("native f16 (default)", &model, &g, device, &inputs, false)?;
    run_benchmark("simulated f16 (forced)", &model, &g, device, &inputs, true)?;
    run_benchmark("CPU f16 (default)", &model, &g, Device::Cpu, &inputs, false)?;

    Ok(())
}
