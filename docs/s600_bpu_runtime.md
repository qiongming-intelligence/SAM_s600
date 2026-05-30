# S600 BPU runtime

SAM_s600 uses S600 BPU through the D-Robotics runtime and converted HBM model files.

Useful runtime checks on the board:

```bash
hb_version
hrt_ucp_monitor -b -d 1000 -e bpu
hrt_model_exec model_info --model_file <model.hbm>
```

`hobot-smi` is not used as a project dependency because it targets PAC/PCIe management rather than local SAM3 BPU utilization.


## Inspecting converted SAM3 parts

Any converted SAM3 HBM partition can be inspected through the shared CLI helper:

```bash
sam3_benchmark --inspect-part models/hbm/sam3_image_encoder.hbm
```

The command prints the model name, compile-time BPU core count, input tensor metadata, and output tensor metadata.


## Loading SAM3 partitions from a manifest

Use `--load-parts` to make a SAM3 CLI load every existing HBM partition referenced by a manifest and report per-part status:

```bash
sam3_benchmark --manifest configs/manifests/sam3_full.yaml --load-parts
```

Use `--require-all` when every partition must be present and loadable:

```bash
sam3_benchmark --manifest configs/manifests/sam3_full.yaml --load-parts --require-all
```
