# SAM3 on S600: end-to-end setup and runbook

This document describes the public, reproducible flow for running SAM3/SAM3.1 on a D-Robotics/Horizon S600 BPU target:

1. prepare the Python export environment;
2. obtain authorized upstream SAM3/SAM3.1 assets;
3. export the default SAM3 full detector to ONNX;
4. optionally cast large outputs to FP16;
5. compile ONNX to HBM with the S600 toolchain;
6. copy the HBM to the S600 target;
7. run model inspection, benchmark, and application smoke tests.

The documented default path keeps the original SAM3 image processing size:

- input image tensor: `1x3x1008x1008`;
- object queries: `200`;
- mask logits: `200x288x288`;
- SAM3/SAM3.1 scope only.

No private machine names, usernames, internal IPs, or local absolute paths are required. Replace placeholders such as `<repo>`, `<toolchain-host>`, `<s600-target>`, and `/path/to/...` with your own environment.

## 1. Repository layout

Important project paths:

```text
<repo>/src/python/sam_s600_tools/        Python export/conversion tools
<repo>/src/python/scripts/               HBM compile and benchmark scripts
<repo>/configs/manifests/                Runtime manifests
<repo>/docs/                             Documentation
<repo>/build/                            Generated artifacts; not committed
<repo>/models/hbm/                       Generated HBM files; not committed
<repo>/models/upstream/                  Authorized upstream weights; not committed
```

Generated assets such as `.onnx`, `.onnx.data`, `.hbm`, checkpoints, and `build/` outputs are intentionally ignored by git.

## 2. Environments

Use separate environments for export, HBM compilation, and target execution.

### 2.1 Python export environment

This environment is used to download authorized assets, load the upstream model, and export ONNX.

Recommended packages:

- Python 3.10 or 3.11;
- PyTorch;
- torchvision;
- transformers with SAM3/SAM3.1 support;
- ONNX;
- ONNX Runtime;
- NumPy;
- safetensors;
- PyYAML;
- Hugging Face Hub and/or ModelScope client.

Example:

```bash
conda create -y -n sam3-export -c conda-forge --override-channels python=3.11 pip
conda install -y -n sam3-export -c conda-forge --override-channels \
  pytorch torchvision onnx onnxruntime numpy safetensors pyyaml \
  huggingface_hub transformers tokenizers accelerate
conda run -n sam3-export python -m pip install modelscope
```

Verify imports:

```bash
conda run -n sam3-export python - <<'PY'
import torch, onnx, onnxruntime, transformers
print("torch", torch.__version__)
print("onnx", onnx.__version__)
print("onnxruntime", onnxruntime.__version__)
print("transformers", transformers.__version__)
PY
```

### 2.2 HBM compilation environment

HBM compilation must run on an x86_64 Linux host with the D-Robotics/OpenExplorer S600 toolchain installed.

Required commands:

```bash
hb_compile --version
python3 --version
```

The compile scripts expect `hb_compile` in `PATH`. If your toolchain is installed in a conda environment or SDK setup script, activate it before running the compile commands.

### 2.3 S600 target runtime environment

Runtime validation and benchmark must run on the S600 target board.

Required command:

```bash
/usr/hobot/bin/hrt_model_exec --help
```

The local C++ runtime binaries are built from this repository. At minimum, the benchmark flow below only requires `hrt_model_exec` and the generated HBM.

## 3. Authorized SAM3/SAM3.1 weights

This project does not redistribute SAM3/SAM3.1 checkpoints. Obtain the upstream assets through an authorized provider and comply with the upstream license.

Example Hugging Face flow:

```bash
cd <repo>
conda run -n sam3-export huggingface-cli login
PYTHONPATH=src/python conda run -n sam3-export \
  python -m sam_s600_tools.download.weights \
  --source huggingface \
  --variant sam3 \
  --out-dir models/upstream
```

Example ModelScope flow:

```bash
cd <repo>
PYTHONPATH=src/python conda run -n sam3-export \
  python -m sam_s600_tools.download.weights \
  --source modelscope \
  --variant sam3 \
  --out-dir models/upstream
```

Use `--variant sam3.1` for SAM3.1.

Expected output location, using the ModelScope example:

```text
models/upstream/modelscope/sam3/
models/upstream/modelscope/sam3.1/
```

These directories are local generated assets and should not be committed.

## 4. Export ONNX

There are two export styles in this repository:

1. partition export, used by the modular C++ runtime;
2. full-detector export, used by the current best S600 performance path.

The current best measured S600 path is the full-detector HBM compiled with two BPU cores.

### 4.1 Generate export contracts

Contracts describe tensor names, shapes, and expected partition interfaces.

```bash
cd <repo>
PYTHONPATH=src/python conda run -n sam3-export \
  python -m sam_s600_tools.export.contract \
  --out-dir build/sam3_export \
  --print-plan
```

### 4.2 Export with the Transformers-format factory

For a Transformers/ModelScope-format upstream directory:

```bash
cd <repo>
PYTHONPATH=src/python conda run -n sam3-export \
  python -m sam_s600_tools.export.onnx \
  --model-factory sam_s600_tools.export.factories.transformers:create_model \
  --model-config models/upstream/modelscope/sam3 \
  --out-dir build/sam3_export \
  --partition detector \
  --image-size 1008 \
  --mask-size 288 \
  --max-objects 200 \
  --dry-run
```

If the dry run resolves the detector module, remove `--dry-run`:

```bash
cd <repo>
PYTHONPATH=src/python conda run -n sam3-export \
  python -m sam_s600_tools.export.onnx \
  --model-factory sam_s600_tools.export.factories.transformers:create_model \
  --model-config models/upstream/modelscope/sam3 \
  --out-dir build/sam3_export \
  --partition detector \
  --image-size 1008 \
  --mask-size 288 \
  --max-objects 200 \
  --device cpu
```

This writes ONNX files under the `onnx_path` recorded in the generated contracts.

### 4.3 Full-detector ONNX used by the optimized HBM path

The optimized HBM compile script expects a full-detector ONNX with this six-input ABI:

```text
pixel_values          1x3x1008x1008  float32
input_ids             1x16           int32
attention_mask        1x16           int32
input_boxes           1x2x4          float32
input_boxes_labels    1x2            int32
geometry_roi_features 2x256x7x7      float32
```

and these outputs:

```text
object_logits         1x200           float32
object_boxes          1x200x4         float32
mask_logits           1x200x288x288   float32 or float16
presence_logits       1x1             float32
semantic_segmentation 1x1x288x288     float32 or float16
```

Use the exported or otherwise validated full-detector ONNX as:

```text
build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full.onnx
build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full.onnx.data
```

If your export path writes a different filename, pass it through `SAM3_FULL_SOURCE_ONNX` in the compile commands below.

Validate the ONNX before HBM compilation:

```bash
cd <repo>
PYTHONPATH=src/python conda run -n sam3-export python - <<'PY'
import onnx
path = "build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full.onnx"
onnx.checker.check_model(path)
print("ONNX checker passed:", path)
PY
```

## 5. Optional output-only FP16 ONNX

The fastest measured default-SAM3 S600 variant keeps internal compute FP32 and casts only large outputs to FP16:

- `mask_logits`;
- `semantic_segmentation`.

This preserves the default input size and output shapes while reducing output bandwidth. It does not change `query=200` or `mask=288`.

```bash
cd <repo>
PYTHONPATH=src/python conda run -n sam3-export \
  python -m sam_s600_tools.convert.output_precision \
  build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full.onnx \
  build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full_output_fp16.onnx \
  --precision fp16 \
  --outputs mask_logits,semantic_segmentation
```

The output ONNX should be used as the HBM compiler input for the best latency variant.

## 6. Compile ONNX to HBM

Run this section on the x86_64 toolchain host, not on the S600 target.

### 6.1 Best measured full-detector variant

Compile the output-only-FP16 ONNX with two BPU cores, O2 optimization, and latency mode:

```bash
cd <repo>
# Activate your S600/OpenExplorer toolchain environment here.

SAM3_FULL_SOURCE_ONNX=build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full_output_fp16.onnx \
SAM3_FULL_OUTPUT_PREFIX=sam3_detector_full_outfp16_core2_O2_latency \
PYTHONPATH=src/python \
src/python/scripts/compile_full_detector_matrix.sh \
  fp32 2 O2 latency \
  models/hbm
```

`fp32` is used here because the graph internals remain FP32; only selected graph outputs are FP16.

Expected output:

```text
models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm
```

Generate a checksum:

```bash
sha256sum models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm \
  > models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm.sha256
```

### 6.2 Pure FP32 reference variant

For a clean FP32 reference:

```bash
cd <repo>
# Activate your S600/OpenExplorer toolchain environment here.

SAM3_FULL_SOURCE_ONNX=build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full.onnx \
PYTHONPATH=src/python \
src/python/scripts/compile_full_detector_matrix.sh \
  fp32 2 O2 latency \
  models/hbm
```

Expected output:

```text
models/hbm/sam3_detector_full_fp32_core2_O2_latency.hbm
```

### 6.3 Knobs that were not beneficial

The following variants were measured or rejected during optimization and are not recommended as defaults:

```text
compile_mode=bandwidth      slower than latency mode
optimize_level=O1           slower than O2
balance_factor=50           slower than default latency mode
balance_factor>100          invalid for the compiler
whole-graph INT8 PTQ        slower than FP32 for this graph
internal selective FP16     can trigger CPU fallback unless carefully revalidated
```

## 7. Copy HBM to the S600 target

Run this from the S600 target or any machine that can reach the toolchain host.

```bash
cd <repo>
mkdir -p models/hbm

rsync -avP --partial --append-verify \
  <toolchain-host>:/path/to/repo/models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm \
  models/hbm/

rsync -avP --partial \
  <toolchain-host>:/path/to/repo/models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm.sha256 \
  models/hbm/
```

Verify checksum:

```bash
cd <repo>
sha256sum models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm
cat models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm.sha256
```

## 8. Inspect and benchmark on S600

Run on the S600 target.

### 8.1 Inspect model ABI

```bash
/usr/hobot/bin/hrt_model_exec model_info \
  --model_file models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm
```

Expected input ABI:

```text
pixel_values          1x3x1008x1008  F32
input_ids             1x16           S32
attention_mask        1x16           S32
input_boxes           1x2x4          F32
input_boxes_labels    1x2            S32
geometry_roi_features 2x256x7x7      F32
```

Expected output ABI for the best variant:

```text
object_logits         1x200           F32
object_boxes          1x200x4         F32
mask_logits           1x200x288x288   F16
presence_logits       1x1             F32
semantic_segmentation 1x1x288x288     F16
```

### 8.2 Prepare benchmark inputs

The benchmark wrapper expects six raw input files in ABI order under a directory such as:

```text
models/hbm/perf_inputs_full_f32/
```

Required files:

```text
pixel_values_f32.bin
input_ids_s32.bin
attention_mask_s32.bin
input_boxes_f32.bin
input_boxes_labels_s32.bin
geometry_roi_features_f32.bin
```

These files must already match the HBM input shapes and dtypes. For real images, preprocess upstream SAM3/SAM3.1 images to `1x3x1008x1008` float32 before writing `pixel_values_f32.bin`.

SAM3 default image preprocessing is:

```text
convert to RGB
resize directly to 1008x1008
rescale by 1/255
normalize with mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
pack as NCHW float32
```

### 8.3 Run benchmark

```bash
cd <repo>
src/python/scripts/bench_full_detector.sh \
  models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm \
  sam3_detector_full_outfp16_core2_O2_latency \
  models/hbm/perf_inputs_full_f32 \
  30
```

The script runs:

- single-core scheduling;
- dual-core scheduling with one thread;
- dual-core scheduling with two threads;
- BPU/CPU profiler summaries.

The expected best measured latency on S600 is approximately:

```text
Average latency: ~286 ms/frame
BPU time:        ~285 ms/frame
CPU time:        ~0.3 ms/frame
FPS:             ~3.5
```

Exact results vary with firmware, SDK/toolchain version, thermal state, input files, and measurement settings.

## 9. End-to-end application smoke run

The current C++ image CLI expects raw preprocessed tensor bytes, not arbitrary JPG/PNG decoding. If you already have a raw `pixel_values` tensor matching the HBM input, run:

```bash
cd <repo>
./build/sam3_benchmark \
  --manifest configs/manifests/sam3_full.yaml \
  --run \
  --image /path/to/preprocessed_pixel_values_f32.bin \
  --text "person"
```

If your build names differ, use the corresponding `sam3_benchmark` binary produced by your CMake build.

Important limitations for the current runtime:

- decoded JPG/PNG preprocessing is not yet implemented in the C++ image processor;
- `--image` is a raw byte-stream input path;
- the raw input must already match the compiled HBM tensor ABI;
- production applications should implement or call the upstream SAM3 preprocessing before invoking the BPU model.

## 10. Build C++ runtime binaries

A typical native build on the S600 target is:

```bash
cd <repo>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

If your SDK requires extra include/library paths, pass them through your toolchain file or CMake cache variables. Keep machine-specific SDK paths out of committed files.

## 11. Troubleshooting

### `hb_compile` not found

Activate the x86_64 S600/OpenExplorer toolchain environment and check:

```bash
which hb_compile
hb_compile --version
```

### HBM runs on one core only

Make sure the HBM was compiled with:

```yaml
compiler_parameters:
  core_num: 2
```

and benchmark with:

```bash
--core_id 1,2 --thread_num 1
```

### Output dtype does not match expected FP16

Inspect the ONNX generated by `output_precision.py` and re-run `hrt_model_exec model_info`. The best variant should show:

```text
mask_logits:           F16
semantic_segmentation: F16
```

### Very large CPU inference time

This usually means a graph region fell back to CPU. Avoid deploying internal selective-FP16 variants unless `profiler.csv` confirms CPU time remains near the FP32 baseline.

### Real image results look distorted

The documented default processor directly resizes input images to `1008x1008`. If your application preserves aspect ratio or pads instead, coordinates and masks will not match the default SAM3 preprocessing path unless postprocessing is adjusted accordingly.

## 12. What to commit

Commit source code, scripts, docs, configs, and manifests.

Do not commit:

```text
*.hbm
*.onnx
*.onnx.data
*.pt
*.pth
*.safetensors
models/hbm/
models/upstream/
build/
```

These are generated artifacts or licensed assets and are ignored by `.gitignore`.
