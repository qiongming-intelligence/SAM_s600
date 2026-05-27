# Models

This directory stores model manifests and local placeholders.

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

Generate export contracts with `tools/export/sam3_export_contract.py`, convert real ONNX exports with `tools/convert/convert_sam3_to_hbm.sh`, and regenerate manifests with `tools/convert/generate_model_manifest.py`.
