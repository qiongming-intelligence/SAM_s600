# SAM3 model export

This project does not redistribute SAM3 checkpoints. Users must obtain authorized SAM3/SAM3.1 weights from the upstream provider and comply with the SAM License.

Current export status: contract and conversion-plan tooling is in place. Real checkpoint tracing and ONNX export still depends on the exact upstream SAM3 module names and exportable forward signatures from the authorized checkpoint package.

## Pipeline

1. Inspect an authorized upstream SAM3/SAM3.1 checkpoint.
2. Generate partition export contracts.
3. Trace/export each partition to ONNX using the upstream Python implementation.
4. Validate ONNX outputs against upstream PyTorch.
5. Convert validated ONNX models to S600 HBM with the D-Robotics toolchain.
6. Generate manifests consumed by the C++ runtime.
7. Validate HBM tensor names/shapes/types with `sam3_* --inspect-part` and runtime smoke runs.

## Export contract generation

Generate contracts for all SAM3/SAM3.1 partitions:

```bash
tools/export/sam3_export_contract.py --out-dir build/sam3_export --print-plan
```

Generate a subset through the per-partition wrappers:

```bash
tools/export/export_sam3_image_encoder.py --out-dir build/sam3_export --print-plan
tools/export/export_sam3_detector.py --out-dir build/sam3_export --print-plan
tools/export/export_sam3_video_tracker.py --out-dir build/sam3_export --print-plan
tools/export/export_sam3_multiplex.py --out-dir build/sam3_export --print-plan
```

Optionally inspect a checkpoint if PyTorch is installed:

```bash
tools/export/sam3_export_contract.py \
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

## HBM conversion

After real ONNX files exist at the `onnx_path` values recorded in the contracts, convert them on an S600 toolchain host:

```bash
tools/convert/convert_sam3_to_hbm.sh build/sam3_export/contracts models/hbm
```

The script expects `hb_mapper` in `PATH` and writes converted files into `models/hbm/`.

Validate a converted HBM file:

```bash
tools/convert/validate_hbm.sh models/hbm/sam3_image_encoder.hbm
build-apps/sam3_image --inspect-part models/hbm/sam3_image_encoder.hbm
```

## Manifest generation

Regenerate manifests from the expected HBM filenames:

```bash
tools/convert/generate_model_manifest.py --hbm-dir models/hbm --out-dir models/manifests
```

Use `--require-existing` to fail if any expected HBM file is missing.

## Next implementation stage

The next non-placeholder step is to add upstream SAM3 module adapters that import the authorized SAM3 Python package, resolve the partition modules from the checkpoint, construct dummy inputs matching each contract, and call `torch.onnx.export` for each partition.
