# Contributing

SAM_s600 is a SAM3/SAM3.1-only S600 edge inference project.

## Scope

Please keep contributions focused on:

- SAM3/SAM3.1 C++ runtime and BPU integration
- SAM3 model export and HBM conversion tooling
- SAM3-only examples, tests, and benchmarks
- Documentation for the S600 SAM3 workflow

Do not add unrelated model families, generic model-zoo examples, or non-SAM3 benchmark results.

## Development

Build the C++ runtime:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Build and run tests:

```bash
cmake -S . -B build-tests -DCMAKE_BUILD_TYPE=Release -DSAM_S600_BUILD_TESTS=ON
cmake --build build-tests -j$(nproc)
ctest --test-dir build-tests --output-on-failure
```

Install Python tooling for local development:

```bash
python3 -m pip install -e src/python
```

## Models and assets

Do not commit upstream checkpoints, ONNX exports, HBM files, or large calibration/demo assets. Checked-in manifests live in `configs/manifests/` and may reference local generated files under `models/hbm/`.
