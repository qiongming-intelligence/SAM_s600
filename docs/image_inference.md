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

The C++ image predictor now validates required SAM3 image partitions, allocates BPU tensor buffers from loaded HBM metadata, and can execute the image encoder when input bytes already match the converted encoder tensors. End-to-end SAM3 object/mask decoding will land after the exported SAM3 detector and mask decoder HBM contracts are fixed.
