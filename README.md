# SAM_s600

High-performance C++ implementation of SAM3/SAM3.1 inference on D-Robotics S600 BPU.

SAM_s600 aims to bring the full capability set of Meta SAM3 to S600-class edge devices: image segmentation, text and visual prompts, interactive segmentation, video object tracking, and SAM3.1 Object Multiplex-style multi-object tracking.

This repository is currently in pre-alpha. The initial codebase establishes the public project structure, C++ runtime interfaces, model conversion workflow, and capability roadmap. SAM3 model execution will land incrementally.

## Scope

This project is only for SAM3/SAM3.1.

Non-goals:

- No generic model zoo
- No non-SAM3 demos or benchmark reports

This repository is scoped exclusively to SAM3/SAM3.1 runtime, tooling, examples, and benchmarks.

## Target capabilities

| Capability | Status |
|---|---|
| SAM3 image text prompt segmentation | Planned |
| SAM3 image point prompt segmentation | Planned |
| SAM3 image box prompt segmentation | Planned |
| SAM3 image mask prompt segmentation | Planned |
| SAM3 image exemplar / visual prompt segmentation | Planned |
| Batched SAM3 image inference | Planned |
| Interactive SAM3 image inference | Planned |
| SAM3 video promptable segmentation | Planned |
| SAM3 video object tracking | Planned |
| SAM3 multi-object video tracking | Planned |
| SAM3.1 Object Multiplex-style tracking | Planned |
| USB/MIPI camera demo | Planned |
| H.264/RTSP video demo | Planned |
| C++ API | In progress |
| Python bindings | Later |
| S600 BPU performance report for SAM3 only | Planned |

See [docs/sam3_capability_matrix.md](docs/sam3_capability_matrix.md) for the detailed roadmap.

## Architecture

SAM_s600 is organized around a C++ runtime with explicit SAM3 module boundaries:

```text
input image/video/prompt
  -> preprocessing
  -> SAM3 tokenizer / text encoder
  -> SAM3 image encoder
  -> geometry / prompt encoder
  -> vision-language fusion / detector
  -> mask decoder
  -> video tracker / multiplex state
  -> postprocess / visualization
```

The BPU integration layer is isolated under `include/sam_s600/bpu` and `src/bpu` so the SAM3 API remains stable while model partitioning evolves.

## Repository layout

```text
include/sam_s600/      Public C++ headers
src/                   Runtime implementation
apps/                  CLI demo and benchmark entry points
tools/export/          SAM3 PyTorch/ONNX export helpers
tools/convert/         S600 HBM conversion helpers
tools/benchmark/       SAM3-only benchmark scripts
models/                Model manifests; no upstream weights committed
docs/                  Architecture, conversion, runtime, and performance docs
examples/              SAM3 usage examples
benchmarks/            SAM3-only benchmark reports
```

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The initial skeleton builds without requiring SAM3 weights. Real inference will require converted `.hbm` model parts generated from authorized SAM3 checkpoints.

## Model weights and license

Meta SAM3 checkpoints are distributed under the SAM License and require access through the official checkpoint host. This repository does not redistribute Meta checkpoints.

Planned workflow:

```text
authorized SAM3 checkpoint
  -> export subgraphs
  -> validate ONNX
  -> convert to S600 HBM
  -> run with SAM_s600 C++ runtime
```

See [docs/license_and_weights.md](docs/license_and_weights.md).

## Development milestones

1. C++ SAM3 runtime interfaces
2. S600 BPU model loader and tensor runtime
3. SAM3 image text-prompt MVP
4. Full image prompt support
5. Video segmentation and tracking
6. SAM3.1 multiplex tracking
7. SAM3-only performance optimization and public demo assets

## Related upstream project

- Meta SAM3: https://github.com/facebookresearch/sam3

## Status

This repository is being built as an open-source edge inference project for SAM3 on S600. APIs and file layout may change before the first tagged release.
