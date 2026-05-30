# SAM3.1 multiplex tracking

SAM3.1 introduces Object Multiplex-style joint multi-object tracking. SAM_s600 exposes this as a dedicated runtime path with explicit multiplex state.

Target CLI:

```bash
sam3_multiplex_video --input video.mp4 --prompts prompts.json
```

Use `--run` to invoke the C++ multiplex predictor from the CLI:

```bash
sam3_multiplex_video --manifest configs/manifests/sam3_multiplex.yaml --input frame0.bin --run
sam3_multiplex_video --manifest configs/manifests/sam3_multiplex.yaml --input raw_frames/ --text "person" --text "ball" --run
```

For the current pre-alpha runtime, `--input` may be a single raw frame or a directory of raw frame files sorted by path. Each raw frame must already match the multiplex detector input tensors. The multiplex detector, tracker, and memory encoder partitions are validated and executed in a minimal shared-state skeleton; prompt demuxing and stable object-ID propagation will land in later stages.

The implementation should prioritize shared state reuse, reduced per-object work, and stable object IDs across frames.
