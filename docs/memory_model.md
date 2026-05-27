# Memory model

S600 BPU workloads use shared system memory regions exposed through HBMEM/ION/CMA-style allocation. SAM_s600 keeps allocation behind `BpuAllocator` so model execution can use BPU-addressable buffers on-device and still build/run with a host fallback on non-S600 machines.

`BpuAllocator` owns buffers through move-only RAII `BpuBuffer` objects. When the Hobot BPU memory SDK is available, allocations can target BPU memory (`hb_bpu_mem_alloc`) or CPU memory registered with the BPU memory subsystem (`hb_bpu_cpumem_alloc`). Buffers expose CPU-accessible data pointers, BPU addresses, physical addresses, per-core IOVA queries, and cache clean/invalidate helpers for cacheable allocations.

Optimization targets:

- preallocated input/output buffers
- prompt embedding cache
- image embedding cache
- video memory state cache
- multiplex object state reuse
- minimal CPU/BPU copy boundaries
