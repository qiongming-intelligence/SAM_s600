# SAM3.1 multiplex tracking

SAM3.1 introduces Object Multiplex-style joint multi-object tracking. SAM_s600 exposes this as a dedicated runtime path with explicit multiplex state.

Target CLI:

```bash
sam3_multiplex_video --input video.mp4 --prompts prompts.json
```

The implementation should prioritize shared state reuse, reduced per-object work, and stable object IDs across frames.
