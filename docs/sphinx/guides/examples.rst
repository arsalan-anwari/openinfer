Examples
========

The `examples/openinfer` directory contains runnable programs that demonstrate
common patterns.

Rust examples (`examples/openinfer`)
------------------------------------

- `mlp_regression`: basic MLP forward pass (matmul + bias + relu).
- `linear_attention`: multi-head linear attention composition with a loop.
- `quantized_linear`: i4 quantized matmul with i32 accumulation.
- `moe_routing`: Mixture-of-Experts routing with `branch` control flow.
- `residual_mlp_stack`: residual MLP stack using `loop` and patterned weights.
- `stability_guard`: `is_finite` guard that branches to a fallback path.
- `streaming_pipeline`: `yield`/`await` streaming pipeline across blocks.
- `online_weight_update`: persistent weight update with `cache.read/write`.
- `kv_cache_decode`: fixed-size KV cache and full table read.
- `cache_window_slice`: cache slice window with explicit start/end indices.

Running examples
----------------

.. code-block:: bash

   cargo run --example mlp_regression

Python `.oinf` examples
-----------------------

The `examples/openinfer-oinf` directory mirrors the Rust examples and shows how
to generate `.oinf` files using Python tooling.

- `mlp_regression_oinf`: MLP forward pass baseline.
- `linear_attention_oinf`: linear attention graph generation.
- `quantized_linear_oinf`: i4 quantized linear layer generation.
- `moe_routing_oinf`: MoE routing with branch control flow.
- `residual_mlp_stack_oinf`: residual MLP stack with patterned weights.
- `stability_guard_oinf`: numerical stability guard with fallback.
- `streaming_pipeline_oinf`: streaming pipeline graph with yield/await.
- `online_weight_update_oinf`: persistent weight update example.
- `kv_cache_decode_oinf`: fixed KV cache read/write.
- `cache_window_slice_oinf`: cache slice window example.
