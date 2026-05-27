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

Use `--run` to invoke the C++ video predictor from the CLI:

```bash
sam3_video --manifest models/manifests/sam3_video.yaml --input frame0.bin --run
sam3_video_tracking --manifest models/manifests/sam3_video.yaml --input raw_frames/ --box 10,20,300,400 --run
```

For the current pre-alpha runtime, `--input` may be a single raw frame or a directory of raw frame files sorted by path. Each raw frame must already match the converted image encoder HBM input tensors. Decoded video demuxing and image-folder preprocessing will be added once the SAM3 video export contract is finalized.

The current pre-alpha CLI validates request wiring and model manifests. The C++ `Sam3VideoPredictor` now validates required SAM3 video partitions (image encoder, detector, mask decoder, memory encoder, tracker) and can run the image encoder on each raw frame whose bytes already match the converted HBM input tensors. Tracker/memory chaining, demuxing, and SAM3.1 multiplex execution will land in following stages.
