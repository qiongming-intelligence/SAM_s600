# SAM3 capability matrix

SAM_s600 targets the original SAM3/SAM3.1 capability set on D-Robotics S600 BPU.

| Area | Capability | Status | Notes |
|---|---|---|---|
| Image | Text prompt segmentation | Planned | Open-vocabulary concept segmentation. |
| Image | Point prompt segmentation | Planned | Interactive SAM-style point prompts. |
| Image | Box prompt segmentation | Planned | Interactive box prompts. |
| Image | Mask prompt segmentation | Planned | Mask refinement prompts. |
| Image | Exemplar / visual prompt segmentation | Planned | Visual concept prompts where supported by upstream SAM3. |
| Image | Batched image inference | Planned | Batch prompt/image execution. |
| Video | Promptable video segmentation | Planned | Text/point/box/mask prompts on video. |
| Video | Object tracking | Planned | Track prompt-selected objects over frames. |
| Video | Multi-object tracking | Planned | Multiple objects and masklets. |
| SAM3.1 | Object Multiplex-style tracking | Planned | Shared state for joint multi-object tracking. |
| Runtime | C++ API | In progress | Initial API skeleton exists. |
| Runtime | S600 BPU subgraph execution | Planned | HBM model parts from exported SAM3 subgraphs. |
| Runtime | Python bindings | Later | Optional wrapper after C++ API stabilizes. |

Only SAM3/SAM3.1 capabilities are tracked here.
