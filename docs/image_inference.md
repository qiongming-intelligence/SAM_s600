# Image inference

Supported CLI request fields are in place for the SAM3 image modes:

- text prompt: `--text "person"`
- point prompt: `--point x,y[,label]`
- box prompt: `--box x0,y0,x1,y1`
- mask prompt: `--mask prompt_mask.png`
- exemplar / visual prompt: `--exemplar exemplar.jpg`

Example commands:

```bash
sam3_image --image input.jpg --text "person" --manifest models/manifests/sam3_image.yaml
sam3_image_interactive --image input.jpg --point 120,240,1 --box 10,20,300,400
sam3_image --image input.jpg --mask prompt_mask.png --exemplar exemplar.jpg
```

Use `--run` to invoke the C++ image predictor from the CLI:

```bash
sam3_image --manifest models/manifests/sam3_image.yaml --image encoder_input.bin --run
```

`encoder_input.bin` is a raw byte stream that must already match the converted image encoder HBM input tensors. Decoded JPG/PNG preprocessing will be added once the SAM3 image encoder export contract is finalized.

The C++ image predictor validates required SAM3 image partitions, allocates BPU tensor buffers from loaded HBM metadata, runs prompt encoders for text and geometry prompts, and chains image encoder, detector, and mask decoder partitions when their exported tensor names/shapes/types line up. End-to-end SAM3 object/mask decoding will land after the detector and mask decoder output contracts are fixed.

