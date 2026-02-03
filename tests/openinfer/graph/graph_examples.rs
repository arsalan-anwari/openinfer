use anyhow::Result;
use openinfer::{fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator};
use std::path::Path;

use crate::common;

#[test]
fn branching_good_sets_condition_true() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/branching_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[B, D];
            }

            constant {
                w: f32[D, D];
            }

            volatile {
                h: f32[B, D];
                cond: bool @init(true);
            }

            block entry {
                op matmul(x, w) >> h;
                op is_finite(h) >> cond;
                branch cond ok bad;
                branch algorithm;
                return;
            }

            block ok {
                op relu(h, alpha=0.0) >> h;
                return;
            }

            block bad {
                op fill(x, value=0.0) >> x;
                op fill(h, value=0.0) >> h;
                return;
            }

            block algorithm {
                op add(h, x) >> h;
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping branching_good on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let b = model.size_of("B")?;
        let d = model.size_of("D")?;
        let input = Random::<f32>::generate_with_seed_opts(
            3,
            (-2.0, 2.0),
            b * d,
            openinfer::TensorOptions {
                shape: Some(vec![b, d]),
                ..openinfer::TensorOptions::default()
            },
        )?;

        insert_executor!(exec, { x: input });
        exec.step()?;

        fetch_executor!(exec, { cond: bool });
        assert!(cond);
    }
    Ok(())
}

#[test]
fn branching_bad_sets_condition_false() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/branching_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[B, D];
            }

            constant {
                w: f32[D, D];
            }

            volatile {
                h: f32[B, D];
                cond: bool @init(false);
            }

            block entry {
                op matmul(x, w) >> h;
                op is_finite(h) >> cond;
                branch cond ok bad;
                branch algorithm;
                return;
            }

            block ok {
                op relu(h, alpha=0.0) >> h;
                return;
            }

            block bad {
                op fill(x, value=0.0) >> x;
                op fill(h, value=0.0) >> h;
                return;
            }

            block algorithm {
                op add(h, x) >> h;
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping branching_bad on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let b = model.size_of("B")?;
        let d = model.size_of("D")?;
        let mut input = Random::<f32>::generate_with_seed_opts(
            3,
            (-2.0, 2.0),
            b * d,
            openinfer::TensorOptions {
                shape: Some(vec![b, d]),
                ..openinfer::TensorOptions::default()
            },
        )?;
        if !input.data.is_empty() {
            input.data[0] = f32::NAN;
        }

        insert_executor!(exec, { x: input });
        exec.step()?;

        fetch_executor!(exec, { cond: bool });
        assert!(!cond);
    }
    Ok(())
}

#[test]
fn cache_scalar_increments() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/cache_scalar_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            volatile {
                out_step: i32;
            }

            persistent {
                step: i32 @init(0);
            }

            block entry {
                cache.read step >> out_step;
                cache.increment step;
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping cache_scalar on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        exec.step()?;
        fetch_executor!(exec, { out_step: i32 });
        assert_eq!(out_step, 0);

        exec.step()?;
        fetch_executor!(exec, { out_step: i32 });
        assert_eq!(out_step, 1);
    }
    Ok(())
}
