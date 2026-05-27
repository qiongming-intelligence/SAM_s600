# Model partitioning

SAM3 is too large and dynamic to treat as one monolithic edge model. SAM_s600 will partition the model around stable inference boundaries:

- image encoder
- text encoder
- geometry encoder
- detector / decoder
- mask decoder
- memory encoder
- video tracker
- multiplex detector/tracker

The first implementation may use CPU fallback for unsupported pieces, but public milestones should move each performance-critical path to BPU where practical.
