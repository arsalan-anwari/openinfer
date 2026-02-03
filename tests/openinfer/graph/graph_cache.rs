use anyhow::Result;
use openinfer::{fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor};
use std::path::Path;

use crate::common;

#[test]
fn cache_decrement_scalar() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/cache_scalar_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            volatile {
                out_step: i32;
            }

            persistent {
                step: i32 @init(5);
            }

            block entry {
                cache.read step >> out_step;
                cache.decrement 2 step;
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping cache_decrement on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        exec.step()?;
        fetch_executor!(exec, { out_step: i32 });
        assert_eq!(out_step, 5);

        exec.step()?;
        fetch_executor!(exec, { out_step: i32 });
        assert_eq!(out_step, 3);
    }
    Ok(())
}

#[test]
fn cache_table_multidim_slice_reads() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/cache_table_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[D];
            }

            volatile {
                out_single: f32[D];
                out_slice: f32[3, D];
            }

            persistent {
                A(i, j): f32[D] @table @fixed(i=2, j=3);
            }

            block entry {
                cache.write x >> A[1, 0];
                cache.write x >> A[1, 1];
                cache.write x >> A[1, 2];
                cache.read A[1, 1] >> out_single;
                cache.read A[1, 0..3] >> out_slice;
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping cache_table_multidim on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let len = model.size_of("D")?;
        let input = Random::<f32>::generate_with_seed(11, (-1.0, 1.0), len)?;
        insert_executor!(exec, { x: input.clone() });
        exec.step()?;

        fetch_executor!(exec, { out_single: Tensor<f32>, out_slice: Tensor<f32> });
        assert_eq!(out_single.data, input.data);
        assert_eq!(out_slice.shape(), &[3, len]);
        for row in 0..3 {
            let start = row * len;
            let end = start + len;
            assert_eq!(&out_slice.data[start..end], &input.data[..]);
        }
    }
    Ok(())
}
