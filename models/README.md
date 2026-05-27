# Models

This directory stores model manifests and local placeholders.

SAM3 checkpoints and converted HBM files are not committed to this repository.

Expected local layout after conversion:

```text
models/hbm/
  sam3_image_encoder.hbm
  sam3_text_encoder.hbm
  sam3_detector.hbm
  sam3_mask_decoder.hbm
  sam3_video_tracker.hbm
  sam3_multiplex_detector.hbm
  sam3_multiplex_tracker.hbm
```
