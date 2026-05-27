#!/usr/bin/env python3
"""SAM3/SAM3.1 export contract generation for S600 conversion."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: list[str]
    source: str


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    upstream_candidates: list[str]
    onnx_name: str
    hbm_name: str
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]


PARTITIONS: dict[str, PartitionSpec] = {
    "image_encoder": PartitionSpec(
        name="image_encoder",
        upstream_candidates=["image_encoder", "vision_encoder", "backbone", "image_backbone"],
        onnx_name="sam3_image_encoder.onnx",
        hbm_name="sam3_image_encoder.hbm",
        inputs=[TensorSpec("image", "uint8", ["1", "H", "W", "C"], "raw_preprocessed_image")],
        outputs=[TensorSpec("image_embeddings", "float16", ["1", "C", "H/stride", "W/stride"], "image_encoder")],
    ),
    "text_encoder": PartitionSpec(
        name="text_encoder",
        upstream_candidates=["text_encoder", "language_encoder", "prompt_encoder.text"],
        onnx_name="sam3_text_encoder.onnx",
        hbm_name="sam3_text_encoder.hbm",
        inputs=[
            TensorSpec("input_ids", "int32", ["1", "T"], "tokenizer"),
            TensorSpec("attention_mask", "int32", ["1", "T"], "tokenizer"),
        ],
        outputs=[TensorSpec("text_embeddings", "float16", ["1", "T", "C"], "text_encoder")],
    ),
    "geometry_encoder": PartitionSpec(
        name="geometry_encoder",
        upstream_candidates=["geometry_encoder", "prompt_encoder", "point_encoder", "box_encoder"],
        onnx_name="sam3_geometry_encoder.onnx",
        hbm_name="sam3_geometry_encoder.hbm",
        inputs=[
            TensorSpec("points", "float32", ["1", "P", "2"], "prompt_points"),
            TensorSpec("point_labels", "int32", ["1", "P"], "prompt_points"),
            TensorSpec("boxes", "float32", ["1", "B", "4"], "prompt_boxes"),
        ],
        outputs=[TensorSpec("geometry_embeddings", "float16", ["1", "G", "C"], "geometry_encoder")],
    ),
    "detector": PartitionSpec(
        name="detector",
        upstream_candidates=["detector", "object_detector", "prompt_detector"],
        onnx_name="sam3_detector.onnx",
        hbm_name="sam3_detector.hbm",
        inputs=[
            TensorSpec("image_embeddings", "float16", ["1", "C", "H/stride", "W/stride"], "image_encoder"),
            TensorSpec("text_embeddings", "float16", ["1", "T", "C"], "text_encoder"),
            TensorSpec("geometry_embeddings", "float16", ["1", "G", "C"], "geometry_encoder"),
        ],
        outputs=[
            TensorSpec("object_logits", "float16", ["1", "N"], "detector"),
            TensorSpec("object_boxes", "float16", ["1", "N", "4"], "detector"),
            TensorSpec("object_tokens", "float16", ["1", "N", "C"], "detector"),
        ],
    ),
    "mask_decoder": PartitionSpec(
        name="mask_decoder",
        upstream_candidates=["mask_decoder", "segmentation_decoder"],
        onnx_name="sam3_mask_decoder.onnx",
        hbm_name="sam3_mask_decoder.hbm",
        inputs=[
            TensorSpec("image_embeddings", "float16", ["1", "C", "H/stride", "W/stride"], "image_encoder"),
            TensorSpec("object_tokens", "float16", ["1", "N", "C"], "detector"),
        ],
        outputs=[
            TensorSpec("mask_logits", "float16", ["1", "N", "Hm", "Wm"], "mask_decoder"),
            TensorSpec("mask_scores", "float16", ["1", "N"], "mask_decoder"),
        ],
    ),
    "memory_encoder": PartitionSpec(
        name="memory_encoder",
        upstream_candidates=["memory_encoder", "video_memory_encoder"],
        onnx_name="sam3_memory_encoder.onnx",
        hbm_name="sam3_memory_encoder.hbm",
        inputs=[
            TensorSpec("image_embeddings", "float16", ["1", "C", "H/stride", "W/stride"], "image_encoder"),
            TensorSpec("object_tokens", "float16", ["1", "N", "C"], "detector"),
            TensorSpec("mask_logits", "float16", ["1", "N", "Hm", "Wm"], "mask_decoder"),
            TensorSpec("track_tokens", "float16", ["1", "N", "C"], "tracker"),
        ],
        outputs=[TensorSpec("memory_tokens", "float16", ["1", "M", "C"], "memory_encoder")],
    ),
    "tracker": PartitionSpec(
        name="tracker",
        upstream_candidates=["tracker", "video_tracker", "track_predictor"],
        onnx_name="sam3_video_tracker.onnx",
        hbm_name="sam3_video_tracker.hbm",
        inputs=[
            TensorSpec("image_embeddings", "float16", ["1", "C", "H/stride", "W/stride"], "image_encoder"),
            TensorSpec("object_tokens", "float16", ["1", "N", "C"], "detector"),
            TensorSpec("memory_tokens", "float16", ["1", "M", "C"], "memory_encoder"),
        ],
        outputs=[TensorSpec("track_tokens", "float16", ["1", "N", "C"], "tracker")],
    ),
    "multiplex_detector": PartitionSpec(
        name="multiplex_detector",
        upstream_candidates=["multiplex_detector", "object_multiplex.detector"],
        onnx_name="sam3_multiplex_detector.onnx",
        hbm_name="sam3_multiplex_detector.hbm",
        inputs=[TensorSpec("multiplex_frame", "uint8", ["1", "H", "W", "C"], "raw_preprocessed_frame")],
        outputs=[TensorSpec("multiplex_object_tokens", "float16", ["1", "N", "C"], "multiplex_detector")],
    ),
    "multiplex_tracker": PartitionSpec(
        name="multiplex_tracker",
        upstream_candidates=["multiplex_tracker", "object_multiplex.tracker"],
        onnx_name="sam3_multiplex_tracker.onnx",
        hbm_name="sam3_multiplex_tracker.hbm",
        inputs=[
            TensorSpec("multiplex_object_tokens", "float16", ["1", "N", "C"], "multiplex_detector"),
            TensorSpec("memory_tokens", "float16", ["1", "M", "C"], "memory_encoder"),
        ],
        outputs=[TensorSpec("multiplex_track_tokens", "float16", ["1", "N", "C"], "multiplex_tracker")],
    ),
}


def selected_partitions(names: Iterable[str]) -> list[PartitionSpec]:
    result = []
    for name in names:
        if name not in PARTITIONS:
            known = ", ".join(sorted(PARTITIONS))
            raise SystemExit(f"unknown SAM3 partition: {name}; known: {known}")
        result.append(PARTITIONS[name])
    return result


def checkpoint_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"checkpoint does not exist: {path}")
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise SystemExit("checkpoint inspection requires PyTorch; install torch or omit --inspect-checkpoint") from error

    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not hasattr(state, "keys"):
        raise SystemExit("checkpoint does not contain a dict-like state_dict")

    keys = [str(key) for key in state.keys()]
    prefixes = Counter(key.split(".", 1)[0] for key in keys)
    return {
        "checkpoint": str(path),
        "parameter_count": len(keys),
        "top_level_prefixes": dict(sorted(prefixes.items())),
        "matched_partitions": {
            name: [candidate for candidate in spec.upstream_candidates if any(candidate in key for key in keys)]
            for name, spec in PARTITIONS.items()
        },
    }


def concrete_shape(shape: list[str], dims: dict[str, int]) -> list[int]:
    values = []
    for item in shape:
        expr = item
        for key, value in sorted(dims.items(), key=lambda pair: len(pair[0]), reverse=True):
            expr = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])", str(value), expr)
        if not re.fullmatch(r"[0-9+\-*/() ]+", expr):
            raise SystemExit(f"shape expression is still symbolic: {item}")
        values.append(int(eval(expr, {"__builtins__": {}}, {})))
    return values


def tensor_contract(tensor: TensorSpec, dims: dict[str, int]) -> dict[str, object]:
    data = asdict(tensor)
    data["concrete_shape"] = concrete_shape(tensor.shape, dims)
    return data


def write_contracts(partitions: list[PartitionSpec], out_dir: Path, checkpoint: Path | None, sam3_repo: Path | None, dims: dict[str, int]) -> None:
    contracts_dir = out_dir / "contracts"
    onnx_dir = out_dir / "onnx"
    hbm_dir = Path("models/hbm")
    contracts_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    index = {
        "format": "sam_s600_export_contract/v1",
        "status": "contract_only",
        "checkpoint": str(checkpoint) if checkpoint else None,
        "sam3_repo": str(sam3_repo) if sam3_repo else None,
        "dimensions": dims,
        "partitions": [],
    }

    for spec in partitions:
        contract = asdict(spec)
        contract["inputs"] = [tensor_contract(tensor, dims) for tensor in spec.inputs]
        contract["outputs"] = [tensor_contract(tensor, dims) for tensor in spec.outputs]
        contract.update(
            {
                "format": "sam_s600_partition_contract/v1",
                "status": "contract_only",
                "dimensions": dims,
                "onnx_path": str(onnx_dir / spec.onnx_name),
                "hbm_path": str(hbm_dir / spec.hbm_name),
                "export_entrypoint": f"tools/export/export_sam3_{spec.name}.py",
                "notes": [
                    "Bind C++ runtime stages by exact tensor names.",
                    "Update dimensions after upstream SAM3 module tracing fixes concrete export sizes.",
                    "Do not commit upstream checkpoints, ONNX files, or HBM files.",
                ],
            }
        )
        path = contracts_dir / f"{spec.name}.json"
        path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        index["partitions"].append({"name": spec.name, "contract": str(path), "onnx": contract["onnx_path"], "hbm": contract["hbm_path"]})

    (out_dir / "export_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_export_plan(partitions: list[PartitionSpec], out_dir: Path) -> None:
    hb_mapper = shutil.which("hb_mapper") or "hb_mapper"
    for spec in partitions:
        onnx_path = out_dir / "onnx" / spec.onnx_name
        hbm_path = Path("models/hbm") / spec.hbm_name
        print(f"[{spec.name}]")
        print(f"  contract: {out_dir / 'contracts' / f'{spec.name}.json'}")
        print(f"  onnx:     {onnx_path}")
        print(f"  hbm:      {hbm_path}")
        print(f"  convert:  {hb_mapper} makertbin --model-type onnx --model {onnx_path} --output-dir {hbm_path.parent}")


def build_parser(default_partitions: list[str] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SAM3/SAM3.1 partition export contracts for S600 conversion.")
    parser.add_argument("--checkpoint", type=Path, help="authorized upstream SAM3/SAM3.1 checkpoint path")
    parser.add_argument("--sam3-repo", type=Path, help="local upstream SAM3 repository checkout")
    parser.add_argument("--out-dir", type=Path, default=Path("build/sam3_export"), help="export work directory")
    parser.add_argument("--partition", action="append", choices=sorted(PARTITIONS), help="partition to include; repeatable")
    parser.add_argument("--image-size", type=int, default=1008, help="square image/frame export size")
    parser.add_argument("--embed-channels", type=int, default=256, help="feature/token channel count")
    parser.add_argument("--image-stride", type=int, default=16, help="image encoder stride")
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--max-geometry-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=256)
    parser.add_argument("--max-memory-tokens", type=int, default=256)
    parser.add_argument("--mask-size", type=int, default=256)
    parser.add_argument("--inspect-checkpoint", action="store_true", help="load checkpoint with PyTorch and write checkpoint_summary.json")
    parser.add_argument("--print-plan", action="store_true", help="print ONNX/HBM conversion plan")
    parser.set_defaults(default_partitions=default_partitions)
    return parser


def export_dims(args: argparse.Namespace) -> dict[str, int]:
    return {
        "H": args.image_size,
        "W": args.image_size,
        "C": args.embed_channels,
        "T": args.max_text_tokens,
        "G": args.max_geometry_tokens,
        "P": args.max_geometry_tokens,
        "B": args.max_geometry_tokens,
        "N": args.max_objects,
        "M": args.max_memory_tokens,
        "Hm": args.mask_size,
        "Wm": args.mask_size,
        "stride": args.image_stride,
    }


def main(default_partitions: list[str] | None = None) -> int:
    parser = build_parser(default_partitions)
    args = parser.parse_args()
    partition_names = args.partition or args.default_partitions or list(PARTITIONS)
    partitions = selected_partitions(partition_names)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_contracts(partitions, args.out_dir, args.checkpoint, args.sam3_repo, export_dims(args))
    if args.inspect_checkpoint:
        if args.checkpoint is None:
            raise SystemExit("--inspect-checkpoint requires --checkpoint")
        summary = checkpoint_summary(args.checkpoint)
        (args.out_dir / "checkpoint_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.print_plan:
        print_export_plan(partitions, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
