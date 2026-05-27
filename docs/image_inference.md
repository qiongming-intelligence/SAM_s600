# Image inference

Planned image modes:

- text prompt
- point prompt
- box prompt
- mask prompt
- exemplar / visual prompt
- batched prompts
- interactive refinement

CLI targets:

```bash
sam3_image --image input.jpg --text "person" --manifest models/manifests/sam3_image.yaml
sam3_image_interactive --image input.jpg --points points.json
```
