use anyhow::{anyhow, Result};
use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor, TensorOptions,
};
use std::path::Path;

use crate::common;

#[test]
fn cache_table_reads_and_resets() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/cache_table_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[D];
            }

            volatile {
                out_full: f32[3, D];
                out_slice: f32[2, D];
                out_reset: f32[3, D];
            }

            persistent {
                A(i): f32[D] @table @fixed(i=3);
            }

            block entry {
                cache.write x >> A[0];
                cache.write x >> A[1];
                cache.write x >> A[2];

                cache.read A[] >> out_full;
                cache.read A[0..2] >> out_slice;

                cache.reset A[1];
                cache.read A[] >> out_reset;
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping cache_table on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let len = model.size_of("D")?;
        let input = Random::<f32>::generate_with_seed(1, (-1.0, 1.0), len)?;
        insert_executor!(exec, { x: input.clone() });
        exec.step()?;

        fetch_executor!(exec, { out_full: Tensor<f32>, out_slice: Tensor<f32>, out_reset: Tensor<f32> });
        assert_eq!(out_full.shape(), &[3, len]);
        assert_eq!(out_slice.shape(), &[2, len]);
        assert_eq!(out_reset.shape(), &[3, len]);

        for row in 0..3 {
            let start = row * len;
            let end = start + len;
            if row == 1 {
                assert!(out_reset.data[start..end].iter().all(|v| *v == 0.0));
            } else {
                assert_eq!(&out_reset.data[start..end], &input.data[..]);
            }
            assert_eq!(&out_full.data[start..end], &input.data[..]);
        }
        assert_eq!(&out_slice.data[..len], &input.data[..]);
        assert_eq!(&out_slice.data[len..], &input.data[..]);
    }
    Ok(())
}

#[test]
fn cache_auto_dim_increments() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/cache_auto_dim_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                bias: f32;
            }

            volatile {
                out_l: f32[D, H];
                out_m: f32[D, H];
                out_n: f32[D, H];
            }

            persistent {
                l_rows: i32 @init(0);
                l_cols: i32 @init(0);
                m_cols: i32 @init(0);
                n_rows: i32 @init(0);
                L(r, c): f32[D, H] @auto_dim(r, c);
                M(r, c): f32[D, H] @auto_dim(r, c);
                N(r, c): f32[D, H] @auto_dim(r, c);
            }

            block entry {
                cache.increment 3 l_rows;
                cache.increment 2 l_cols;
                cache.increment 5 m_cols;
                cache.increment 5 n_rows;

                cache.read L[l_rows, l_cols] >> out_l;
                cache.read M[0, m_cols] >> out_m;
                cache.read N[n_rows, 0] >> out_n;

                op add(out_l, bias) >> out_l;
                op add(out_m, bias) >> out_m;
                op add(out_n, bias) >> out_n;

                cache.write out_l >> L[l_rows, l_cols];
                cache.write out_m >> M[0, m_cols];
                cache.write out_n >> N[n_rows, 0];
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping cache_auto_dim on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        insert_executor!(exec, { bias: 1.0f32 });
        exec.step()?;
        fetch_executor!(exec, { out_l: Tensor<f32>, out_m: Tensor<f32>, out_n: Tensor<f32> });
        assert_eq!(out_l.shape().len(), 2);
        assert_eq!(out_m.shape().len(), 2);
        assert_eq!(out_n.shape().len(), 2);
        assert!(out_l.data.iter().all(|v| v.is_finite()));
        assert!(out_m.data.iter().all(|v| v.is_finite()));
        assert!(out_n.data.iter().all(|v| v.is_finite()));

        insert_executor!(exec, { bias: 1.0f32 });
        exec.step()?;
        fetch_executor!(exec, { out_l: Tensor<f32>, out_m: Tensor<f32>, out_n: Tensor<f32> });
        assert_eq!(out_l.shape().len(), 2);
        assert_eq!(out_m.shape().len(), 2);
        assert_eq!(out_n.shape().len(), 2);
    }
    Ok(())
}

#[test]
fn cache_fixed_limit_errors() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/cache_auto_dim_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                bias: f32;
            }

            volatile {
                out: f32[D, H];
            }

            persistent {
                rows: i32 @init(0);
                cols: i32 @init(0);
                M(r, c): f32[D, H] @auto_dim(r, c) @fixed(r=4, c=4);
            }

            block entry {
                cache.increment 5 rows;
                cache.increment 5 cols;
                cache.read M[rows, cols] >> out;
                op add(out, bias) >> out;
                cache.write out >> M[rows, cols];
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping cache_fixed_limit on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        insert_executor!(exec, { bias: 1.0f32 });
        let result = exec.step();
        assert!(result.is_err());
    }
    Ok(())
}

#[test]
fn cache_weight_update_accumulates() -> Result<()> {
    for device in common::test_targets() {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../res/models/cache_weight_update_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[D];
                delta: f32[D];
            }

            volatile {
                tmp: f32[D];
                out: f32[D];
            }

            persistent {
                W: f32[D] @init(0.0);
            }

            block entry {
                cache.read W >> tmp;
                op add(tmp, x) >> tmp;
                op add(tmp, delta) >> tmp;
                cache.write tmp >> W;
                cache.read W >> out;
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping cache_weight_update on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let len = model.size_of("D")?;
        let input = Random::<f32>::generate_with_seed(3, (-1.0, 1.0), len)?;
        let delta = Random::<f32>::generate_with_seed(9, (-0.1, 0.1), len)?;

        insert_executor!(exec, { x: input.clone(), delta: delta.clone() });
        exec.step()?;
        fetch_executor!(exec, { out: Tensor<f32> });
        let expected_step1: Vec<f32> = input
            .data
            .iter()
            .zip(delta.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        assert_vec_close(&out.data, &expected_step1, 1e-5)?;

        insert_executor!(exec, { x: input, delta: delta });
        exec.step()?;
        fetch_executor!(exec, { out: Tensor<f32> });
        let expected_step2: Vec<f32> = expected_step1.iter().map(|v| v * 2.0).collect();
        assert_vec_close(&out.data, &expected_step2, 1e-5)?;
    }
    Ok(())
}

#[test]
fn loop_matches_cpu_reference() -> Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/loop_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[D, 3*D];
        }

        volatile {
            y: f32[D, 3*D] @init(0.0);
        }

        constant {
            QKV(layer, head): f32[D, 3*D] @pattern("attn.{head}.qkv.{layer}");
        }

        block entry {
            op add(x, QKV[0, 0]) >> y;

            loop layers (l in 0..num_layers) {
                loop heads (h in 0..num_heads) {
                    op add(y, QKV[l, h]) >> y;
                }
            }

            return;
        }
    };

    let d = model.size_of("D")?;
    let len = d * 3 * d;
    let input = Random::<f32>::generate_with_seed_opts(
        42,
        (-1.0, 1.0),
        len,
        TensorOptions {
            shape: Some(vec![d, 3 * d]),
            ..TensorOptions::default()
        },
    )?;

    let cpu_sim = Simulator::new(&model, &g, openinfer::Device::Cpu)?;
    let mut cpu_exec = cpu_sim.make_executor()?;
    insert_executor!(cpu_exec, { x: input.clone() });
    cpu_exec.step()?;
    fetch_executor!(cpu_exec, { y: Tensor<f32> });
    let expected = y;

    for device in common::test_targets() {
        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping loop on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;
        insert_executor!(exec, { x: input.clone() });
        exec.step()?;
        fetch_executor!(exec, { y: Tensor<f32> });
        if device == openinfer::Device::Cpu {
            assert_eq!(y.data, expected.data);
        } else {
            let actual = openinfer::TensorValue::F32(y);
            let expected = openinfer::TensorValue::F32(expected.clone());
            common::assert_tensor_close(&actual, &expected, device)?;
        }
    }
    Ok(())
}

#[test]
fn yield_updates_value() -> Result<()> {
    for device in common::test_targets() {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../res/models/yield_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[B, D];
            }

            constant {
                w: f32[D, D];
                bias: f32[D];
            }

            volatile {
                h: f32[B, D];
                h2: f32[B, D];
            }

            block entry {
                op matmul(x, w) >> h;
                yield x;

                op relu(h, alpha=0.0, clamp_max=6.0) >> h;

                await x;
                return;
            }

            block writer {
                await x;
                op add(x, bias) >> x;
                yield x;
            }

            block reader {
                await x;
                op relu(x, alpha=0.0, clamp_max=6.0) >> h2;
                yield x;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping yield on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let b = model.size_of("B")?;
        let d = model.size_of("D")?;
        let input = Random::<f32>::generate_with_seed_opts(
            0,
            (-1.0, 1.0),
            b * d,
            TensorOptions {
                shape: Some(vec![b, d]),
                ..TensorOptions::default()
            },
        )?;
        insert_executor!(exec, { x: input.clone() });
        exec.step()?;

        fetch_executor!(exec, { x: Tensor<f32> });
        let bias = model.load_tensor("bias")?;
        let bias = bias.as_f32()?.data.clone();
        let mut expected = Vec::with_capacity(x.data.len());
        for row in 0..b {
            for col in 0..d {
                expected.push(input.data[row * d + col] + bias[col]);
            }
        }
        assert_eq!(x.data, expected);
    }
    Ok(())
}

#[test]
fn prefix_table_matches_model_constants() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/prefix_table_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[D];
            }

            volatile {
                y: f32[D];
                W(l): f32[D] @pattern("W.{l}");
            }

            constant {
                QKV(layer, head): f32[D] @pattern("QKV.{layer}.{head}");
            }

            block entry {
                op add(x, W[0]) >> y;
                op add(y, W[1]) >> y;
                op add(y, W[2]) >> y;
                op add(y, W[3]) >> y;
                op add(y, W[4]) >> y;
                op add(y, W[5]) >> y;
                op add(y, W[6]) >> y;
                op add(y, W[7]) >> y;
                op add(y, W[8]) >> y;
                op add(y, W[9]) >> y;
                op add(y, W[10]) >> y;
                op add(y, QKV[0, 0]) >> y;
                op add(y, QKV[1, 2]) >> y;
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping prefix_table on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let len = model.size_of("D")?;
        let input = Random::<f32>::generate_with_seed(7, (-1.0, 1.0), len)?;
        insert_executor!(exec, { x: input.clone() });
        exec.step()?;

        fetch_executor!(exec, { y: Tensor<f32> });
        let expected = openinfer::TensorValue::F32(y);

        let sim_cpu = Simulator::new(&model, &g, openinfer::Device::Cpu)?;
        let mut exec_cpu = sim_cpu.make_executor()?;
        insert_executor!(exec_cpu, { x: input });
        exec_cpu.step()?;
        fetch_executor!(exec_cpu, { y: Tensor<f32> });
        let actual = openinfer::TensorValue::F32(y);
        common::assert_tensor_close(&actual, &expected, device)?;
    }
    Ok(())
}

#[test]
fn reference_model_matches_expected() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/reference_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[B];
            }

            volatile {
                state: f32[B] @ref("state.0");
                out: f32[B];
            }

            constant {
                weight: f32[B] @ref("weight.0");
                bias: f32 @ref("bias.0");
            }

            block entry {
                assign t0: f32[B];
                op add(x, weight) >> t0;
                op add(t0, state) >> out;
                op add(out, bias) >> out;
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping reference on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let len = model.size_of("B")?;
        let input = Random::<f32>::generate_with_seed(7, (-2.0, 2.0), len)?;
        insert_executor!(exec, { x: input.clone() });
        exec.step()?;

        fetch_executor!(exec, { out: Tensor<f32> });
        let expected = openinfer::TensorValue::F32(out);

        let sim_cpu = Simulator::new(&model, &g, openinfer::Device::Cpu)?;
        let mut exec_cpu = sim_cpu.make_executor()?;
        insert_executor!(exec_cpu, { x: input });
        exec_cpu.step()?;
        fetch_executor!(exec_cpu, { out: Tensor<f32> });
        let actual = openinfer::TensorValue::F32(out);
        common::assert_tensor_close(&actual, &expected, device)?;
    }
    Ok(())
}

#[test]
fn attrs_from_model_applies_values() -> Result<()> {
    for device in common::test_targets() {
        let model_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/attrs_from_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let g = graph! {
            dynamic {
                x: f32[B];
            }

            volatile {
                y_model: f32[B];
                y_hard: f32[B];
                cast_model: i32[B];
                cast_hard: i32[B];
            }

            constant {
                alpha: f32;
                clamp_max: f32;
            }

            block entry {
                op relu(x, alpha=alpha, clamp_max=clamp_max) >> y_model;
                op relu(x, alpha=0.2, clamp_max=2.5) >> y_hard;
                op cast(y_model, to=i32, rounding_mode=rounding_mode, saturate=true) >> cast_model;
                op cast(y_hard, to=i32, rounding_mode="nearest", saturate=false) >> cast_hard;
                return;
            }
        };

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping attrs_from_model on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let mut exec = sim.make_executor()?;

        let len = model.size_of("B")?;
        let input = Random::<f32>::generate_with_seed(0, (-3.0, 3.0), len)?;
        insert_executor!(exec, { x: input.clone() });
        exec.step()?;

        fetch_executor!(exec, { y_model: Tensor<f32>, y_hard: Tensor<f32>, alpha: f32, clamp_max: f32 });
        let rounding_mode = model
            .load_metadata_string("rounding_mode")?
            .unwrap_or_default();
        assert!(!rounding_mode.is_empty());

        let mut expected_model = Vec::with_capacity(len);
        let mut expected_hard = Vec::with_capacity(len);
        for value in &input.data {
            let neg = alpha * *value;
            let pos = value.min(clamp_max);
            expected_model.push(if *value < 0.0 { neg } else { pos });
            let hard_neg = 0.2 * *value;
            let hard_pos = value.min(2.5);
            expected_hard.push(if *value < 0.0 { hard_neg } else { hard_pos });
        }
        assert_eq!(y_model.data, expected_model);
        assert_eq!(y_hard.data, expected_hard);
    }
    Ok(())
}

#[test]
fn serialize_roundtrip_executes() -> Result<()> {
    for device in common::test_targets() {
        let g = graph! {
            dynamic {
                x: f32[B];
            }

            volatile {
                a: f32[B];
                y: f32[B] @init(5.0);
            }

            block entry {
                assign t0: f32[B];
                op add(x, a) >> t0;
                op mul(y, t0) >> y;
                return;
            }
        };

        let json = openinfer::GraphSerialize::json(&g)?;
        let g2 = openinfer::GraphDeserialize::from_json(json)?;

        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../res/models/minimal_model.oinf");
        let model = ModelLoader::open(model_path)?;

        let sim = match Simulator::new(&model, &g, device) {
            Ok(sim) => sim,
            Err(err) => {
                if device == openinfer::Device::Vulkan {
                    eprintln!("Skipping serialize_roundtrip on {:?}: {}", device, err);
                    continue;
                }
                return Err(err);
            }
        };
        let sim2 = Simulator::new(&model, &g2, device)?;
        let mut exec = sim.make_executor()?;
        let mut exec2 = sim2.make_executor()?;

        let len = model.size_of("B")?;
        let input = Random::<f32>::generate_with_seed(0, (-10.0, 10.0), len)?;
        insert_executor!(exec, { x: input.clone() });
        insert_executor!(exec2, { x: input });
        exec.step()?;
        exec2.step()?;

        fetch_executor!(exec, { y: Tensor<f32> });
        let y_ref = y.data.clone();
        fetch_executor!(exec2, { y: Tensor<f32> });
        assert_eq!(y.data, y_ref);
    }
    Ok(())
}

#[test]
fn trace_emits_events() -> Result<()> {
    let model_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/models/minimal_model.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B];
        }

        volatile {
            a: f32[B];
            y: f32[B] @init(5.0);
        }

        block entry {
            assign t0: f32[B];
            op add(x, a) >> t0;
            op mul(y, t0) >> y;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, openinfer::Device::Cpu)?.with_trace().with_timer();
    let mut exec = sim.make_executor()?;

    let len = model.size_of("B")?;
    let input = Random::<f32>::generate_with_seed(0, (-10.0, 10.0), len)?;
    insert_executor!(exec, { x: input });
    exec.step()?;
    let trace = exec.trace();
    if trace.is_empty() {
        return Err(anyhow!("trace should contain events"));
    }
    Ok(())
}

fn assert_vec_close(actual: &[f32], expected: &[f32], tol: f32) -> Result<()> {
    if actual.len() != expected.len() {
        return Err(anyhow!(
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        ));
    }
    for (idx, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        if (*a - *b).abs() > tol {
            return Err(anyhow!("value mismatch at {}: {} vs {}", idx, a, b));
        }
    }
    Ok(())
}
