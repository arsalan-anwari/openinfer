use anyhow::Result;
use openinfer::{fetch_executor, graph, insert_executor, GraphDeserialize, GraphSerialize, ModelLoader, Random, Simulator, Tensor};
use std::path::Path;

use crate::common;

#[test]
fn serialize_roundtrip_complex_graph() -> Result<()> {
    for device in common::test_targets() {
        let g = graph! {
            dynamic {
                x: f32[B];
            }

            volatile {
                y: f32[B] @init(0.0);
                out_step: i32;
            }

            persistent {
                step: i32 @init(0);
            }

            block entry {
                cache.read step >> out_step;
                cache.increment 1 step;
                loop i (idx in 0..2) {
                    op add(y, x) >> y;
                }
                return;
            }
        };

        let json = GraphSerialize::json(&g)?;
        let g2 = GraphDeserialize::from_json(json)?;

        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/minimal_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping serde_roundtrip on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let sim2 = Simulator::new(&model, &g2, device)?;
        let mut exec = sim.make_executor()?;
        let mut exec2 = sim2.make_executor()?;

        let len = model.size_of("B")?;
        let input = Random::<f32>::generate_with_seed(9, (-1.0, 1.0), len)?;
        insert_executor!(exec, { x: input.clone() });
        insert_executor!(exec2, { x: input });
        exec.step()?;
        exec2.step()?;

        fetch_executor!(exec, { y: Tensor<f32>, out_step: i32 });
        let y_ref = y.data.clone();
        let step_ref = out_step;
        fetch_executor!(exec2, { y: Tensor<f32>, out_step: i32 });
        assert_eq!(y.data, y_ref);
        assert_eq!(out_step, step_ref);
    }
    Ok(())
}

#[test]
fn invalid_branch_target_errors() -> Result<()> {
    let model_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/minimal_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B];
        }

        volatile {
            y: f32[B];
            cond: bool @init(true);
        }

        block entry {
            branch cond ok missing_block;
            return;
        }

        block ok {
            op add(x, x) >> y;
            return;
        }
    };

    let result = Simulator::new(&model, &g, openinfer::Device::Cpu);
    assert!(result.is_err());
    Ok(())
}
