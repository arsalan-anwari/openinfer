use anyhow::Result;
use openinfer::{fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor};
use std::path::Path;

use crate::common;

#[test]
fn branch_unconditional_executes_block() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/minimal_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[B];
            }

            volatile {
                y: f32[B] @init(0.0);
            }

            block entry {
                branch algorithm;
                return;
            }

            block algorithm {
                op add(x, x) >> y;
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping branch_unconditional on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let len = model.size_of("B")?;
        let input = Random::<f32>::generate_with_seed(2, (-1.0, 1.0), len)?;
        insert_executor!(exec, { x: input.clone() });
        exec.step()?;
        fetch_executor!(exec, { y: Tensor<f32> });
        let expected: Vec<f32> = input.data.iter().map(|v| v + v).collect();
        assert_eq!(y.data, expected);
    }
    Ok(())
}

#[test]
fn loop_literal_bounds_accumulates() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/minimal_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[B];
            }

            volatile {
                y: f32[B] @init(0.0);
            }

            block entry {
                loop i (idx in 0..3) {
                    op add(y, x) >> y;
                }
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping loop_literal on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let len = model.size_of("B")?;
        let input = Random::<f32>::generate_with_seed(4, (-1.0, 1.0), len)?;
        insert_executor!(exec, { x: input.clone() });
        exec.step()?;
        fetch_executor!(exec, { y: Tensor<f32> });
        let expected: Vec<f32> = input.data.iter().map(|v| v * 3.0).collect();
        assert_eq!(y.data, expected);
    }
    Ok(())
}

#[test]
fn yield_await_multiple_vars() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/minimal_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[B];
                z: f32[B];
                bias_x: f32[B];
                bias_z: f32[B];
            }

            block entry {
                yield x, z;
                await x, z;
                return;
            }

            block writer {
                await x, z;
                op add(x, bias_x) >> x;
                op add(z, bias_z) >> z;
                yield x, z;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping yield_await_multi on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let len = model.size_of("B")?;
        let input_x = Random::<f32>::generate_with_seed(5, (-1.0, 1.0), len)?;
        let input_z = Random::<f32>::generate_with_seed(6, (-1.0, 1.0), len)?;
        let bias_x = Random::<f32>::generate_with_seed(7, (-0.5, 0.5), len)?;
        let bias_z = Random::<f32>::generate_with_seed(8, (-0.5, 0.5), len)?;
        insert_executor!(
            exec,
            {
                x: input_x.clone(),
                z: input_z.clone(),
                bias_x: bias_x.clone(),
                bias_z: bias_z.clone()
            }
        );
        exec.step()?;

        fetch_executor!(exec, { x: Tensor<f32>, z: Tensor<f32> });
        let actual_x = openinfer::TensorValue::F32(x);
        let actual_z = openinfer::TensorValue::F32(z);

        let sim_cpu = Simulator::new(&model, &g, openinfer::Device::Cpu)?;
        let mut exec_cpu = sim_cpu.make_executor()?;
        insert_executor!(
            exec_cpu,
            {
                x: input_x,
                z: input_z,
                bias_x: bias_x,
                bias_z: bias_z
            }
        );
        exec_cpu.step()?;
        fetch_executor!(exec_cpu, { x: Tensor<f32>, z: Tensor<f32> });
        let expected_x = openinfer::TensorValue::F32(x);
        let expected_z = openinfer::TensorValue::F32(z);

        common::assert_tensor_close(&actual_x, &expected_x, device)?;
        common::assert_tensor_close(&actual_z, &expected_z, device)?;
    }
    Ok(())
}

#[test]
fn transfer_node_moves_value() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/minimal_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[B];
            }

            volatile {
                y: f32[B];
            }

            block entry {
                transfer x >> y;
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping transfer on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let len = model.size_of("B")?;
        let input = Random::<f32>::generate_with_seed(7, (-1.0, 1.0), len)?;
        insert_executor!(exec, { x: input.clone() });
        exec.step()?;
        fetch_executor!(exec, { y: Tensor<f32> });
        assert_eq!(y.data, input.data);
    }
    Ok(())
}
