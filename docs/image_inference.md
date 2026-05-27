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

The current pre-alpha CLI validates request wiring and model manifests. Real inference will land after SAM3 subgraph export and S600 HBM conversion.
