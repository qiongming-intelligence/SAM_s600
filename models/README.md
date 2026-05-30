# Models

This directory stores local generated model files and upstream weights. Checked-in manifests live under `configs/manifests/`.

SAM3 checkpoints, ONNX exports, and converted HBM files are not committed to this repository.

Expected local layout after conversion:

```text
models/hbm/
  sam3_image_encoder.hbm
  sam3_text_encoder.hbm
  sam3_geometry_encoder.hbm
  sam3_detector.hbm
  sam3_mask_decoder.hbm
  sam3_memory_encoder.hbm
  sam3_video_tracker.hbm
  sam3_multiplex_detector.hbm
  sam3_multiplex_tracker.hbm
```

Generate export contracts with `python3 -m sam_s600_tools.export.contract`, convert real ONNX exports with `src/python/scripts/compile_hbm.sh`, and regenerate manifests with `python3 -m sam_s600_tools.manifest.generate`.
