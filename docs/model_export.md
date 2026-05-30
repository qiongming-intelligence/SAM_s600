# SAM3 model export

This project does not redistribute SAM3 checkpoints. Users must obtain authorized SAM3/SAM3.1 weights from the upstream provider and comply with the SAM License.

Current export status: contract and conversion-plan tooling is in place, and the `sam3-export` conda environment has been verified with PyTorch, ONNX, ONNX Runtime, Transformers, and Hugging Face Hub on linux-aarch64. Real checkpoint tracing and ONNX export still requires authorized access to the gated official SAM3/SAM3.1 weights.

## Pipeline

1. Inspect an authorized upstream SAM3/SAM3.1 checkpoint.
2. Generate partition export contracts.
3. Trace/export each partition to ONNX using the upstream Python implementation.
4. Validate ONNX outputs against upstream PyTorch.
5. Convert validated ONNX models to S600 HBM with the D-Robotics toolchain.
6. Generate manifests consumed by the C++ runtime.
7. Validate HBM tensor names/shapes/types with `sam3_* --inspect-part` and runtime smoke runs.

## Export environment

Create and verify the export environment:

```bash
conda create -y -n sam3-export -c conda-forge --override-channels python=3.11 pip
conda install -y -n sam3-export -c conda-forge --override-channels \
  pytorch torchvision onnx onnxruntime numpy safetensors pyyaml huggingface_hub transformers tokenizers accelerate
conda run -n sam3-export python -m pip install modelscope
conda run -n sam3-export python - <<'PY'
import torch, onnx, onnxruntime, transformers
print(torch.__version__, onnx.__version__, onnxruntime.__version__, transformers.__version__)
PY
```

This repository was verified with `torch 2.6.0`, `onnx 1.17.0`, `onnxruntime 1.26.0`, `transformers 5.9.0`, and `modelscope 1.37.1` on linux-aarch64.

## Official gated weights

The official model repos are gated:

- `facebook/sam3`
- `facebook/sam3.1`

Accept the upstream license and authenticate first:

```bash
conda run -n sam3-export huggingface-cli login
```

Then download the authorized assets from Hugging Face:

```bash
conda run -n sam3-export python3 -m sam_s600_tools.download.weights --source huggingface --variant sam3 --out-dir models/upstream
conda run -n sam3-export python3 -m sam_s600_tools.download.weights --source huggingface --variant sam3.1 --out-dir models/upstream
```

If Hugging Face access is gated for the current token, ModelScope can mirror the official `facebook/sam3` and `facebook/sam3.1` assets:

```bash
conda run -n sam3-export python3 -m sam_s600_tools.download.weights --source modelscope --variant sam3 --out-dir models/upstream
conda run -n sam3-export python3 -m sam_s600_tools.download.weights --source modelscope --variant sam3.1 --out-dir models/upstream
```

Without an authorized Hugging Face token, Hugging Face returns `GatedRepoError 401`. ModelScope download was verified for `facebook/sam3` and `facebook/sam3.1` on this machine.

## Export contract generation

Generate contracts for all SAM3/SAM3.1 partitions:

```bash
python3 -m sam_s600_tools.export.contract --out-dir build/sam3_export --print-plan
```

Generate a subset through the per-partition wrappers:

```bash
python3 -m sam_s600_tools.export.image_encoder --out-dir build/sam3_export --print-plan
python3 -m sam_s600_tools.export.detector --out-dir build/sam3_export --print-plan
python3 -m sam_s600_tools.export.video_tracker --out-dir build/sam3_export --print-plan
python3 -m sam_s600_tools.export.multiplex --out-dir build/sam3_export --print-plan
```

Optionally inspect a checkpoint if PyTorch is installed:

```bash
python3 -m sam_s600_tools.export.contract \
  --checkpoint /path/to/authorized_sam3_checkpoint.pt \
  --inspect-checkpoint \
  --out-dir build/sam3_export
```

This writes `checkpoint_summary.json` with top-level checkpoint prefixes and candidate partition matches. It does not export weights by itself.

## Partition contracts

Contracts are written under `build/sam3_export/contracts/`. They define the tensor names the C++ runtime expects to bind between partitions:

- `image_encoder`
- `text_encoder`
- `geometry_encoder`
- `detector`
- `mask_decoder`
- `memory_encoder`
- `tracker`
- `multiplex_detector`
- `multiplex_tracker`

The shape entries are symbolic until the upstream SAM3 export path fixes concrete dimensions. The exported ONNX graph must preserve these tensor names or the C++ runtime will reject the chain with a missing upstream binding error.

## ONNX export adapter

Once the authorized upstream SAM3 Python package and checkpoint are available, provide a model factory callable that returns the upstream model object:

```python
# my_sam3_factory.py
def create_model(checkpoint=None, model_config=None, device="cpu"):
    # Import the authorized upstream SAM3 package here.
    # Build the model, load checkpoint weights, move to device, and return it.
    ...
```

Run the adapter:

```bash
PYTHONPATH=/path/to/upstream_sam3:$PYTHONPATH \
  python3 -m sam_s600_tools.export.onnx \
  --model-factory my_sam3_factory:create_model \
  --checkpoint /path/to/authorized_sam3_checkpoint.pt \
  --model-config /path/to/upstream_config.yaml \
  --sam3-repo /path/to/upstream_sam3 \
  --out-dir build/sam3_export \
  --partition image_encoder \
  --partition detector
```

The adapter resolves each partition from the returned model using candidates recorded in the contract, such as `image_encoder`, `vision_encoder`, `detector_model`, `mask_decoder`, `memory_encoder`, `tracker_model`, `multiplex_detector`, and `multiplex_tracker`. Use `--dry-run` first to check that the factory and partition names resolve before exporting ONNX.

For the official Transformers-format ModelScope download, use the built-in factory:

```bash
PYTHONPATH=src/python \
  conda run -n sam3-export python -m sam_s600_tools.export.onnx \
  --model-factory sam_s600_tools.export.factories.transformers:create_model \
  --model-config models/upstream/modelscope/sam3 \
  --out-dir build/sam3_export \
  --partition image_encoder \
  --partition detector \
  --dry-run
```

The default concrete export dimensions are conservative placeholders: image size `1008`, embedding channels `256`, stride `16`, max text/geometry/object/memory tokens `256`, and mask size `256`. Override these with `--image-size`, `--embed-channels`, `--image-stride`, `--max-text-tokens`, `--max-geometry-tokens`, `--max-objects`, `--max-memory-tokens`, and `--mask-size` once the upstream SAM3 export contract is fixed. With the ModelScope `facebook/sam3` download, `image_encoder`, `text_encoder`, `geometry_encoder`, `detector`, `mask_decoder`, and `memory_encoder` ONNX export and ONNX checker validation have been verified. The main Transformers `tracker_model` entrypoint is stateful and requires an `inference_session` object, so it still needs a Tensor-only adapter before ONNX export.

## HBM conversion

After real ONNX files exist at the `onnx_path` values recorded in the contracts, convert them on an S600 toolchain host:

```bash
src/python/scripts/compile_hbm.sh build/sam3_export/contracts models/hbm
```

The script expects `hb_mapper` in `PATH` and writes converted files into `models/hbm/`.

Validate a converted HBM file:

```bash
src/python/scripts/validate_hbm.sh models/hbm/sam3_image_encoder.hbm
build/sam3_image --inspect-part models/hbm/sam3_image_encoder.hbm
```

## Manifest generation

Regenerate manifests from the expected HBM filenames:

```bash
python3 -m sam_s600_tools.manifest.generate --hbm-dir models/hbm --out-dir configs/manifests
```

Use `--require-existing` to fail if any expected HBM file is missing.

## Next implementation stage

The next step is to replace the user-supplied factory shim with built-in adapters for the exact authorized SAM3/SAM3.1 upstream package layout once the module names, preprocessing code, and checkpoint format are available locally. After that, validate each ONNX partition numerically against the upstream PyTorch outputs before HBM conversion.
