# Testing and Tools

This page covers how to run tests and how to load the helper aliases in `.oirc`.

## Run tests

Use the repo script:

```bash
./scripts/run_tests.sh
```

Setup + build helpers:
```bash
./scripts/setup_all.sh
./scripts/sync_models.sh
./scripts/build_all.sh
```

Common options:

```bash
./scripts/run_tests.sh --list
./scripts/run_tests.sh --target=cpu
./scripts/run_tests.sh --target=vulkan --features=vulkan
./scripts/run_tests.sh --target=all --features=vulkan
./scripts/run_tests.sh --test-filter openinfer-simulator::ops_misc
./scripts/run_tests.sh --test-filter openinfer-dsl::parse_tests
./scripts/run_tests.sh --test-filter openinfer-oinf::test_common.TestCommon.test_align_up
```

Notes:

- The script runs NumPy baselines before Rust tests.
- `--target` controls the device target; Vulkan requires `--features=vulkan`.
- `--list` prints test names with prefixes for filtering.

## Source `.oirc`

The `.oirc` file defines shell aliases for common tools and scripts:

```bash
source ./.oirc
```

It adds:

- `dataclass_to_oinf` and `verify_oinf` helpers
- Aliases for scripts in `scripts/` (e.g., `run_tests`)
