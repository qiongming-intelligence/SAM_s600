# Video inference

Supported CLI request fields are in place for SAM3 video modes:

- video file or image folder: `--input path`
- RTSP stream: `--url rtsp://...`
- camera device: `--camera device`
- text prompt: `--text "player in red"`
- point prompt: `--point x,y[,label]`
- box prompt: `--box x0,y0,x1,y1`
- mask prompt: `--mask prompt_mask.png`
- exemplar / visual prompt: `--exemplar exemplar.jpg`

Example commands:

```bash
sam3_video --input video.mp4 --text "player in red"
sam3_video_tracking --input frames/ --box 10,20,300,400
sam3_multiplex_video --input video.mp4 --text "person" --text "ball"
sam3_rtsp --url rtsp://example/stream --text "car"
sam3_camera --camera /dev/video0 --point 320,240,1
```

The current pre-alpha CLI validates request wiring and model manifests. Real inference will land after SAM3 video and multiplex subgraph conversion.
