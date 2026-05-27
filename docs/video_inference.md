# Video inference

Planned video modes:

- video text prompt segmentation
- first-frame interactive prompts
- object tracking
- multi-object masklet output
- H.264 file input
- RTSP input
- camera input

CLI targets:

```bash
sam3_video --input video.mp4 --text "player in red"
sam3_video_tracking --input frames/ --box 10,20,300,400
```
