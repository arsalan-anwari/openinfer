# Supported Ops

This document lists currently supported ops and their backend coverage.

## Backend overview

- **CPU:** scalar Rust kernels.
- **CPU (AVX/AVX2):** SIMD kernels for x86_64 when enabled.
- **Vulkan:** compute kernels compiled from Slang.

## Adding custom ops

This is the minimal checklist to add a new op end-to-end.

### CPU

1. Add the op to `openinfer/src/graph.rs`:
   - `OpKind` variant and `as_str()` mapping.
   - Add any new attributes to `OpAttrs` if needed.
2. Add a CPU implementation under `openinfer/src/ops/cpu/<op>/`:
   - `mod.rs`, `<op>.rs`, `registry.rs`.
   - Optional inplace kernels: `<op>_inplace.rs` + `registry_inplace.rs`.
3. Register the op in the CPU registry:
   - `openinfer/src/ops/cpu/registry.rs` for out-of-place.
   - `openinfer/src/ops/cpu/registry_inplace.rs` via the op module.
4. If the op supports broadcasting, update the policy in
   `openinfer/src/ops/registry.rs` (`broadcast_policy`).

### Vulkan

1. Add the op under `openinfer/src/ops/vulkan/<op>/`:
   - `mod.rs`, `registry.rs`, and `<op>.slang`.
2. Add an entry to `openinfer/src/ops/vulkan/shaders.json` so `build.rs`
   compiles and embeds the SPIR-V. (Broadcast is a backend-only exception.)
3. Implement target selection in `openinfer/src/ops/vulkan/<op>/mod.rs`
   and add it to `openinfer/src/ops/vulkan/mod.rs` dispatch.
4. Register the Vulkan kernel in `openinfer/src/ops/vulkan/<op>/registry.rs`.
5. If you add an inplace variant, wire it in
   `openinfer/src/ops/vulkan/<op>/registry_inplace.rs`.
6. Ensure dtype constraints are enforced in the Vulkan backend
   (`openinfer/src/backend/vulkan/mod.rs`).

## Op coverage

The tables below list support by device. Each row includes dtype coverage,
attributes, and behavior notes (broadcasting, output shape, or special cases).

---

## CPU

<table>
  <thead>
    <tr>
      <th>Op name</th>
      <th>Dtype support</th>
      <th>Op attributes</th>
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>add</code></td>
      <td>
        i8, i16, i32, i64<br>
        u8, u16, u32, u64<br>
        f16, f32, f64<br>
        bool, bitset
      </td>
      <td>None</td>
      <td>Elementwise add. Supports broadcasting on CPU.</td>
    </tr>
    <tr>
      <td><code>mul</code></td>
      <td>
        i8, i16, i32, i64<br>
        u8, u16, u32, u64<br>
        f16, f32, f64<br>
        bool, bitset
      </td>
      <td>None</td>
      <td>Elementwise multiply. Supports broadcasting on CPU.</td>
    </tr>
    <tr>
      <td><code>abs</code></td>
      <td>
        i8, i16, i32, i64<br>
        u8, u16, u32, u64<br>
        f16, f32, f64<br>
        bool, bitset
      </td>
      <td>None</td>
      <td>Absolute value. Unsigned, bool, and bitset act as identity.</td>
    </tr>
    <tr>
      <td><code>relu</code></td>
      <td>
        i8, i16, i32, i64<br>
        u8, u16, u32, u64<br>
        f16, f32, f64<br>
        bool, bitset
      </td>
      <td>
        <code>negative_slope</code> (float)<br>
        <code>clamp_max</code> (float)
      </td>
      <td>
        Leaky ReLU: <code>x &gt;= 0 ? x : x * negative_slope</code>,
        then clamped to <code>clamp_max</code>.
      </td>
    </tr>
    <tr>
      <td><code>matmul</code></td>
      <td>
        i8, i16, i32, i64<br>
        u8, u16, u32, u64<br>
        f16, f32, f64<br>
        bool, bitset
      </td>
      <td>None</td>
      <td>2D matrix multiply. Inner dimensions must match.</td>
    </tr>
    <tr>
      <td><code>is_finite</code></td>
      <td>
        all signed<br>
        all unsigned<br>
        f16, f32, f64<br>
        bool, bitset<br>
        Output: bool scalar
      </td>
      <td>None</td>
      <td>
        Returns <code>true</code> if all elements are finite (float types).
        Non-float dtypes always return <code>true</code>.
      </td>
    </tr>
    <tr>
      <td><code>fill</code></td>
      <td>
        i8, i16, i32, i64<br>
        u8, u16, u32, u64<br>
        f16, f32, f64<br>
        bool, bitset
      </td>
      <td><code>value</code> (typed literal)</td>
      <td>
        Fills output with <code>value</code> (no NaN check).
        <code>value</code> must match input dtype.
      </td>
    </tr>
  </tbody>
</table>

---

## CPU (AVX)

<table>
  <thead>
    <tr>
      <th>Op name</th>
      <th>Dtype support</th>
      <th>Op attributes</th>
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>add</code></td>
      <td>Same as CPU (f16 and bitset fall back to scalar)</td>
      <td>None</td>
      <td>Elementwise add. Supports broadcasting on CPU.</td>
    </tr>
    <tr>
      <td><code>mul</code></td>
      <td>Same as CPU (f16 and bitset fall back to scalar)</td>
      <td>None</td>
      <td>Elementwise multiply. Supports broadcasting on CPU.</td>
    </tr>
    <tr>
      <td><code>abs</code></td>
      <td>Same as CPU (f16 and bitset fall back to scalar)</td>
      <td>None</td>
      <td>Absolute value. Unsigned/bool/bitset act as identity.</td>
    </tr>
    <tr>
      <td><code>relu</code></td>
      <td>Same as CPU (f16 and bitset fall back to scalar)</td>
      <td>
        <code>negative_slope</code> (float)<br>
        <code>clamp_max</code> (float)
      </td>
      <td>Leaky ReLU with clamp.</td>
    </tr>
    <tr>
      <td><code>matmul</code></td>
      <td>Same as CPU</td>
      <td>None</td>
      <td>2D matrix multiply. Inner dimensions must match.</td>
    </tr>
    <tr>
      <td><code>is_finite</code></td>
      <td>Same as CPU</td>
      <td>None</td>
      <td>Returns <code>true</code> if all elements are finite.</td>
    </tr>
    <tr>
      <td><code>fill</code></td>
      <td>Same as CPU</td>
      <td><code>value</code> (typed literal)</td>
      <td>Fills output with <code>value</code> (no NaN check).</td>
    </tr>
  </tbody>
</table>

---

## CPU (AVX2)

<table>
  <thead>
    <tr>
      <th>Op name</th>
      <th>Dtype support</th>
      <th>Op attributes</th>
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>add</code></td>
      <td>Same as CPU (f16 and bitset fall back to scalar)</td>
      <td>None</td>
      <td>Elementwise add. Supports broadcasting on CPU.</td>
    </tr>
    <tr>
      <td><code>mul</code></td>
      <td>Same as CPU (f16 and bitset fall back to scalar)</td>
      <td>None</td>
      <td>Elementwise multiply. Supports broadcasting on CPU.</td>
    </tr>
    <tr>
      <td><code>abs</code></td>
      <td>Same as CPU (f16 and bitset fall back to scalar)</td>
      <td>None</td>
      <td>Absolute value. Unsigned/bool/bitset act as identity.</td>
    </tr>
    <tr>
      <td><code>relu</code></td>
      <td>Same as CPU (f16 and bitset fall back to scalar)</td>
      <td>
        <code>negative_slope</code> (float)<br>
        <code>clamp_max</code> (float)
      </td>
      <td>Leaky ReLU with clamp.</td>
    </tr>
    <tr>
      <td><code>matmul</code></td>
      <td>Same as CPU</td>
      <td>None</td>
      <td>2D matrix multiply. Inner dimensions must match.</td>
    </tr>
    <tr>
      <td><code>is_finite</code></td>
      <td>Same as CPU</td>
      <td>None</td>
      <td>Returns <code>true</code> if all elements are finite.</td>
    </tr>
    <tr>
      <td><code>fill</code></td>
      <td>Same as CPU</td>
      <td><code>value</code> (typed literal)</td>
      <td>Fills output with <code>value</code> (no NaN check).</td>
    </tr>
  </tbody>
</table>

---

## Vulkan

<table>
  <thead>
    <tr>
      <th>Op name</th>
      <th>Dtype support</th>
      <th>Op attributes</th>
      <th>Explanation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>add</code></td>
      <td>
        i8, i16, i32, i64<br>
        u8, u16, u32, u64<br>
        f32<br>
        bool
      </td>
      <td>None</td>
      <td>Elementwise add. Supports broadcasting.</td>
    </tr>
    <tr>
      <td><code>mul</code></td>
      <td>
        i8, i16, i32, i64<br>
        u8, u16, u32, u64<br>
        f32<br>
        bool
      </td>
      <td>None</td>
      <td>Elementwise multiply. Supports broadcasting.</td>
    </tr>
    <tr>
      <td><code>abs</code></td>
      <td>
        i8, i16, i32, i64<br>
        u8, u16, u32, u64<br>
        f32<br>
        bool
      </td>
      <td>None</td>
      <td>Absolute value. Unsigned/bool are identity when enabled.</td>
    </tr>
    <tr>
      <td><code>relu</code></td>
      <td>
        i8, i16, i32, i64<br>
        u8, u16, u32, u64<br>
        f32<br>
        bool
      </td>
      <td>
        <code>negative_slope</code> (float)<br>
        <code>clamp_max</code> (float)
      </td>
      <td>Leaky ReLU with clamp.</td>
    </tr>
    <tr>
      <td><code>matmul</code></td>
      <td>
        i8, i16, i32, i64<br>
        u8, u16, u32, u64<br>
        f32<br>
        bool
      </td>
      <td>None</td>
      <td>2D matrix multiply. Inner dimensions must match.</td>
    </tr>
    <tr>
      <td><code>is_finite</code></td>
      <td>
        all signed<br>
        all unsigned<br>
        f32<br>
        bool<br>
        Output: bool scalar
      </td>
      <td>None</td>
      <td>
        Returns <code>true</code> if all elements are finite (float).
        Non-float dtypes return <code>true</code>.
      </td>
    </tr>
    <tr>
      <td><code>fill</code></td>
      <td>
        i8, i16, i32, i64<br>
        u8, u16, u32, u64<br>
        f32<br>
        bool
      </td>
      <td><code>value</code> (typed literal)</td>
      <td>
        Fills output with <code>value</code> (no NaN check).
        <code>value</code> must match input dtype.
      </td>
    </tr>
  </tbody>
</table>

---

## Notes

- Vulkan dtype support is also constrained by `openinfer/src/backend/vulkan/mod.rs`.
- If an op lists a dtype but the GPU lacks the required feature, Vulkan will
  return an error at runtime.