# SAM3 model export

This project does not redistribute SAM3 checkpoints. Users must obtain authorized SAM3/SAM3.1 weights from the upstream provider and comply with the SAM License.

Planned export stages:

1. Load upstream SAM3 checkpoint in Python.
2. Split into stable inference subgraphs.
3. Export each subgraph to ONNX.
4. Validate ONNX outputs against upstream PyTorch.
5. Convert validated ONNX models to S600 HBM.
6. Generate a manifest consumed by the C++ runtime.

Subgraph boundaries are defined in `models/manifests/*.yaml`.
