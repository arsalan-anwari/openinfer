# Contributing to openinfer: HAL, Drivers & Stability Focus

This project is early-stage and highly experimental, and I’m especially looking for contributors interested in:

* hardware abstraction layers (HALs)
* device drivers / backends
* kernel bring-up
* runtime integration
* correctness testing & fuzzing
* performance validation
* CI for heterogeneous hardware

If that’s your jam, welcome 👋

---

## 🎯 What Kinds of Contributors I’m Looking For

Right now I’m most interested in people who want to work on:

### 🔌 Device Backends & HAL

* CPU backends (x86 / ARM)
* CUDA / ROCm / Metal / Vulkan / OpenCL
* Edge devices (Jetson, Raspberry Pi, etc.)
* NPUs / accelerators
* memory models & allocators
* async execution models
* device discovery & fallback paths

### 🧪 Testing & Debugging

* cross-device correctness suites
* regression tests
* kernel-level tests
* fuzzers / stress tests
* race detection & memory leak hunting
* determinism testing
* performance baselines

### ⚙️ Tooling & Infra

* CI pipelines for GPU / Metal / ARM
* emulator-based testing
* golden-output harnesses
* benchmarking frameworks
* build-system integration

---

## 🚧 Ground Rules

To keep development sane and avoid breaking main:

### ❌ Do NOT push directly to `main`

* `main` is always expected to build and pass tests.
* No direct commits except by maintainers.

### ✅ Development happens on `develop`

* New work should branch from `develop`.
* Submit PRs **into `develop`**, not `main`.

Example:

```
feature/cuda-hal
bugfix/cpu-align-crash
test/metal-backend-ci
```

---

## 🔍 Pull Request Expectations

PRs should:

* focus on one logical change
* include tests when relevant
* not silently change APIs
* document backend assumptions
* list hardware tested on
* include before/after perf numbers when applicable
* explain any known limitations

For HAL or driver PRs, please include:

* target device + OS
* driver/runtime versions
* how to reproduce
* what parts of the HAL are implemented
* what is stubbed or unsupported

---

## 👀 Reviews & Merging

* All PRs require review before merge.
* I may request changes — that’s normal 🙂
* Large architectural changes should be discussed first.
* Force-push only on your own PR branch, never shared branches.

Only maintainers merge into `main`.

---

## 🧠 Experimental Code Policy

It’s totally fine to submit:

* incomplete backends
* proof-of-concept drivers
* experimental kernels

…but:

* clearly mark them as **experimental**
* keep them behind feature flags
* avoid breaking existing backends
* don’t degrade baseline CPU paths

---

## 🐛 Bug-Fix Contributions

Bugfix PRs are extremely welcome.

Please include:

* minimal repro
* root cause (if known)
* test added to prevent regression
* devices affected

---

## 📌 Communication

For anything large:

* open a Discussion or Issue first
* outline the design
* describe trade-offs
* call out risks

This is especially important for HAL changes or new backend types.

---

## ❤️ Final Note

This project is still forming, and I’m intentionally trying to keep:

* the HAL clean
* device drivers modular
* correctness non-negotiable
* performance measurable

If you’re excited about systems-level ML infra, compilers, runtimes, or hardware enablement, you’re exactly who I want here.

Thanks for checking out the project 🙌
