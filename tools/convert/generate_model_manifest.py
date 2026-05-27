#!/usr/bin/env python3
"""Generate SAM3 model manifests for converted S600 HBM files."""

from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_PARTS = ["image_encoder", "text_encoder", "geometry_encoder", "detector", "mask_decoder"]
VIDEO_PARTS = IMAGE_PARTS + ["memory_encoder", "tracker"]
MULTIPLEX_PARTS = ["multiplex_detector", "multiplex_tracker", "memory_encoder"]
HBM_NAMES = {
    "image_encoder": "sam3_image_encoder.hbm",
    "text_encoder": "sam3_text_encoder.hbm",
    "geometry_encoder": "sam3_geometry_encoder.hbm",
    "detector": "sam3_detector.hbm",
    "mask_decoder": "sam3_mask_decoder.hbm",
    "memory_encoder": "sam3_memory_encoder.hbm",
    "tracker": "sam3_video_tracker.hbm",
    "multiplex_detector": "sam3_multiplex_detector.hbm",
    "multiplex_tracker": "sam3_multiplex_tracker.hbm",
}


def write_manifest(path: Path, name: str, parts: list[str], hbm_dir: Path) -> None:
    lines = [f"name: {name}", "version: pre-alpha", "parts:"]
    for part in parts:
        hbm_path = hbm_dir / HBM_NAMES[part]
        lines.append(f"  {part}: {hbm_path.as_posix()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SAM3/SAM3.1 manifests from converted HBM filenames.")
    parser.add_argument("--hbm-dir", type=Path, default=Path("models/hbm"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/manifests"))
    parser.add_argument("--require-existing", action="store_true")
    args = parser.parse_args()

    if args.require_existing:
        missing = [name for name in HBM_NAMES.values() if not (args.hbm_dir / name).exists()]
        if missing:
            raise SystemExit("missing HBM files: " + ", ".join(missing))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.out_dir / "sam3_image.yaml", "sam3_image", IMAGE_PARTS, args.hbm_dir)
    write_manifest(args.out_dir / "sam3_video.yaml", "sam3_video", VIDEO_PARTS, args.hbm_dir)
    write_manifest(args.out_dir / "sam3_multiplex.yaml", "sam3_multiplex", MULTIPLEX_PARTS, args.hbm_dir)
    write_manifest(args.out_dir / "sam3_full.yaml", "sam3_full", VIDEO_PARTS + ["multiplex_detector", "multiplex_tracker"], args.hbm_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
