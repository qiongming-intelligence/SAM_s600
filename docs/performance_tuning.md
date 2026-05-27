# Performance tuning

SAM_s600 benchmarks are SAM3-only.

Planned optimization areas:

- static-shape subgraphs where possible
- BPU core scheduling
- HBMEM buffer reuse
- image/text embedding caches
- batched prompt execution
- asynchronous decode/preprocess/inference/postprocess pipeline
- CPU NEON postprocess for masks, boxes, and visualization
