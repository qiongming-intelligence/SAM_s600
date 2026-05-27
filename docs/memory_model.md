# Memory model

S600 BPU workloads use shared system memory regions exposed through HBMEM/ION/CMA-style allocation. SAM_s600 keeps allocation behind `BpuAllocator` so the implementation can evolve from host-backed placeholders to S600 zero-copy buffers.

Optimization targets:

- preallocated input/output buffers
- prompt embedding cache
- image embedding cache
- video memory state cache
- multiplex object state reuse
- minimal CPU/BPU copy boundaries
