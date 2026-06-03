# SAM_s600

High-performance SAM3/SAM3.1 inference tooling and C++ runtime for the D-Robotics/Horizon S600 BPU.

This repository focuses on one goal: **run SAM3/SAM3.1 on S600 as fast and reproducibly as possible**. It contains:

- Python tooling to export/convert SAM3 models;
- ONNX-to-HBM compile scripts for the S600 toolchain;
- local S600 benchmark wrappers;
- C++ runtime scaffolding for SAM3 image/video/prompt pipelines;
- public documentation for the default SAM3 `1008x1008` deployment flow.

This is a SAM3-only project. It does not contain YOLO/ResNet/model-zoo benchmarks, and it does not redistribute upstream SAM3 weights.

## Current status

The current best measured path is a **full-detector SAM3 HBM** compiled from ONNX with:

```text
input size:       1x3x1008x1008
object queries:   200
mask logits:      200x288x288
BPU cores:        2
compile mode:     latency
optimize level:   O2
large outputs:    mask_logits and semantic_segmentation as FP16
```

Best measured latency on S600:

```text
~286 ms/frame, ~3.5 FPS
```

The C++ runtime is still evolving. The benchmark flow is usable today with raw preprocessed tensor inputs and generated HBM files. Full JPG/PNG/video decode and production-grade preprocessing in C++ are still work in progress.

## Performance

Measured on S600 with SAM3 default `1008x1008` input, `query=200`, `mask=288`, fixed six-input full-detector ABI, and `hrt_model_exec` profiling.

| Variant | Latency | BPU time | CPU time | Notes |
|---|---:|---:|---:|---|
| output-only FP16, core2 O2 latency | **~285.7 ms** | ~285.2 ms | ~0.3 ms | Current fastest measured variant |
| FP32, core2 O2 latency | ~286.1 ms | ~285.5 ms | ~0.3 ms | Clean FP32 reference, nearly tied |
| FP32, core2 O2 bandwidth | ~292.5 ms | ~291.9 ms | ~0.3 ms | Slower than latency mode |
| FP32, core2 O1 latency | ~296.3 ms | ~295.8 ms | ~0.3 ms | Slower than O2 |
| FP32, core2 O2, balance_factor=50 | ~308.7 ms | ~308.1 ms | ~0.3 ms | Slower |
| Original FP32 single-core baseline | ~395 ms | ~394 ms | ~0.3 ms | Older baseline |
| Whole-graph INT8 PTQ | ~433 ms | ~432 ms | ~0.3 ms | Slower than FP32 on this graph |
| LLM dynamic-quant HBM | ~1013 ms or worse | high | may fallback | Not current performance path |
| Internal selective FP16 mask-tail | unusable | ~284 ms | huge fallback | CPU fallback, not deployable |

Key takeaway:

- The major win is **compiling the full detector with `core_num=2`** so a single inference uses both S600 BPU cores.
- Output-only FP16 gives a tiny additional improvement by reducing large output bandwidth.
- INT8 PTQ and internal FP16 were not beneficial for this SAM3 full-detector graph on S600.

Exact numbers vary with SDK/toolchain version, firmware, thermal state, and inputs.

## Default SAM3 image size

Original SAM3/SAM3.1 processor configs use a default image size of:

```text
1008x1008
```

The default image path is:

```text
input image, any resolution
  -> RGB
  -> resize directly to 1008x1008
  -> rescale by 1/255
  -> normalize with mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
  -> NCHW float32 tensor, shape 1x3x1008x1008
```

Non-square images are resized to a square. The benchmark path expects the final preprocessed tensor, not a raw JPG/PNG.

## Repository layout

```text
src/csrc/              C++ runtime, public headers, CLI, and binaries
src/python/            SAM3 export, conversion, download, compile, and benchmark tooling
src/python/scripts/    HBM compile and S600 benchmark scripts
configs/manifests/     Runtime manifests
docs/                  Architecture, export, runtime, and performance docs
tests/csrc/            C++ smoke tests
models/                Local generated assets; not committed
build/                 Build/export outputs; not committed
```

Generated models and licensed assets are ignored by git:

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

## Requirements

Use three environments.

### 1. Python export environment

Used for authorized weight download, PyTorch/Transformers model loading, ONNX export, and ONNX graph conversion.

Typical packages:

```text
python 3.10/3.11
pytorch
torchvision
transformers with SAM3/SAM3.1 support
onnx
onnxruntime
numpy
safetensors
pyyaml
huggingface_hub and/or modelscope
```

Example:

```bash
conda create -y -n sam3-export -c conda-forge --override-channels python=3.11 pip
conda install -y -n sam3-export -c conda-forge --override-channels \
  pytorch torchvision onnx onnxruntime numpy safetensors pyyaml \
  huggingface_hub transformers tokenizers accelerate
conda run -n sam3-export python -m pip install modelscope
```

### 2. HBM compile environment

Used for ONNX-to-HBM conversion. This must be an x86_64 Linux machine with the D-Robotics/OpenExplorer S600 toolchain.

Required command:

```bash
hb_compile --version
```

### 3. S600 runtime environment

Used to inspect, benchmark, and run HBM files on the target board.

Required command:

```bash
/usr/hobot/bin/hrt_model_exec --help
```

## Quick start: benchmark the best full-detector HBM

If you already have a compiled HBM and matching six input `.bin` files on the S600 target:

```bash
cd <repo>

src/python/scripts/bench_full_detector.sh \
  models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm \
  sam3_detector_full_outfp16_core2_O2_latency \
  models/hbm/perf_inputs_full_f32 \
  30
```

The input directory must contain:

```text
pixel_values_f32.bin
input_ids_s32.bin
attention_mask_s32.bin
input_boxes_f32.bin
input_boxes_labels_s32.bin
geometry_roi_features_f32.bin
```

The HBM input ABI is:

```text
pixel_values          1x3x1008x1008  F32
input_ids             1x16           S32
attention_mask        1x16           S32
input_boxes           1x2x4          F32
input_boxes_labels    1x2            S32
geometry_roi_features 2x256x7x7      F32
```

For the best output-only FP16 variant, expected outputs are:

```text
object_logits         1x200           F32
object_boxes          1x200x4         F32
mask_logits           1x200x288x288   F16
presence_logits       1x1             F32
semantic_segmentation 1x1x288x288     F16
```

## End-to-end flow from weights to S600 benchmark

The full reproducible flow is documented in [docs/README.md](docs/README.md). The short version is below.

### Step 1: obtain authorized SAM3/SAM3.1 assets

This repository does not redistribute SAM3 checkpoints. Obtain authorized upstream assets and follow the upstream license.

Example ModelScope download:

```bash
cd <repo>
PYTHONPATH=src/python conda run -n sam3-export \
  python -m sam_s600_tools.download.weights \
  --source modelscope \
  --variant sam3 \
  --out-dir models/upstream
```

Use `--variant sam3.1` for SAM3.1.

### Step 2: export ONNX

Generate contracts:

```bash
cd <repo>
PYTHONPATH=src/python conda run -n sam3-export \
  python -m sam_s600_tools.export.contract \
  --out-dir build/sam3_export \
  --print-plan
```

Export with a Transformers/ModelScope-format factory:

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

The optimized compile script expects a full-detector ONNX with the six-input ABI described above. If your ONNX is written to a custom path, pass it through `SAM3_FULL_SOURCE_ONNX` during compilation.

### Step 3: cast large outputs to FP16

This keeps internal compute FP32 and casts only the large mask/segmentation outputs:

```bash
cd <repo>
PYTHONPATH=src/python conda run -n sam3-export \
  python -m sam_s600_tools.convert.output_precision \
  build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full.onnx \
  build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full_output_fp16.onnx \
  --precision fp16 \
  --outputs mask_logits,semantic_segmentation
```

### Step 4: compile ONNX to HBM

Run on the x86_64 S600 toolchain host:

```bash
cd <repo>
# Activate your S600/OpenExplorer toolchain environment here.

SAM3_FULL_SOURCE_ONNX=build/sam3_modelscope_export/onnx/sam3_detector_hbdk_full_output_fp16.onnx \
SAM3_FULL_OUTPUT_PREFIX=sam3_detector_full_outfp16_core2_O2_latency \
PYTHONPATH=src/python \
src/python/scripts/compile_full_detector_matrix.sh \
  fp32 2 O2 latency \
  models/hbm

sha256sum models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm \
  > models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm.sha256
```

`fp32` is used here because the graph internals are still FP32; only selected graph outputs are FP16.

### Step 5: copy HBM to the S600 target

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

### Step 6: inspect and benchmark on S600

```bash
/usr/hobot/bin/hrt_model_exec model_info \
  --model_file models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm

src/python/scripts/bench_full_detector.sh \
  models/hbm/sam3_detector_full_outfp16_core2_O2_latency.hbm \
  sam3_detector_full_outfp16_core2_O2_latency \
  models/hbm/perf_inputs_full_f32 \
  30
```

## C++ runtime build

Native build:

```bash
cd <repo>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Smoke run with a raw preprocessed tensor:

```bash
cd <repo>
./build/sam3_benchmark \
  --manifest configs/manifests/sam3_full.yaml \
  --run \
  --image /path/to/preprocessed_pixel_values_f32.bin \
  --text "person"
```

Current limitation: `--image` is a raw byte-stream path that must already match the compiled HBM input tensor. Decoded JPG/PNG preprocessing is not yet implemented in the C++ runtime.

## Important limitations

- SAM3/SAM3.1 weights are not included.
- Generated ONNX/HBM files are not included.
- Current best benchmark path uses raw preprocessed tensors, not direct JPG/PNG input.
- C++ end-to-end preprocessing and postprocessing are still evolving.
- Internal FP16 and INT8 variants must be revalidated with `profiler.csv`; CPU fallback can make them much slower.

## Documentation

- [docs/README.md](docs/README.md): full end-to-end runbook
- [docs/model_export.md](docs/model_export.md): ONNX export details
- [docs/image_inference.md](docs/image_inference.md): image CLI/runtime notes
- [docs/s600_bpu_runtime.md](docs/s600_bpu_runtime.md): S600 runtime integration
- [docs/performance_tuning.md](docs/performance_tuning.md): performance notes
- [docs/license_and_weights.md](docs/license_and_weights.md): weight/license policy

## Scope

This repository is exclusively scoped to SAM3/SAM3.1 on S600.

Non-goals:

- no generic model zoo;
- no YOLO/ResNet content or benchmarks;
- no redistribution of upstream SAM3/SAM3.1 weights.

## Upstream

- Meta SAM3: <https://github.com/facebookresearch/sam3>

## License

See repository license files and [docs/license_and_weights.md](docs/license_and_weights.md). Upstream SAM3/SAM3.1 weights are governed by their own license and access terms.
