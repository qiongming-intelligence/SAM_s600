# Architecture

SAM_s600 is a SAM3/SAM3.1-only C++ runtime for D-Robotics S600 BPU.

## Runtime layers

```text
apps / examples
  -> SAM3 public C++ API
  -> SAM3 module runtime
  -> BPU model execution layer
  -> S600 HBM / HBMEM buffers
```

## SAM3 module boundaries

The project mirrors upstream SAM3 at module boundaries rather than copying Python runtime structure directly:

- tokenizer
- text encoder
- image encoder
- geometry / prompt encoder
- vision-language fusion
- detector
- mask decoder
- memory encoder
- video tracker
- SAM3.1 multiplex detector/tracker

This lets each subgraph be exported, converted, benchmarked, and optimized independently for S600.

## Execution model

Image inference starts with cached prompt and image embeddings where possible. Video inference keeps tracker and memory state explicit so it can later map to HBMEM-backed buffers and multiplex execution.
