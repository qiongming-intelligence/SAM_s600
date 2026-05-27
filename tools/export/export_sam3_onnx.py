#!/usr/bin/env python3
"""Export SAM3/SAM3.1 partitions to ONNX through an authorized upstream adapter."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from sam3_export_contract import PARTITIONS, export_dims, selected_partitions, write_contracts

DTYPES = {
    "uint8": "uint8",
    "int32": "int32",
    "float16": "float16",
    "float32": "float32",
}


def make_contract_module(torch: Any, module: Any, output_count: int) -> Any:
    class _Wrapper(torch.nn.Module):
        def __init__(self, wrapped: Any, count: int):
            super().__init__()
            self.wrapped = wrapped
            self.count = count

        def forward(self, *args: Any) -> Any:
            output = self.wrapped(*args)
            if isinstance(output, dict):
                values = list(output.values())
            elif hasattr(output, "to_tuple"):
                values = list(output.to_tuple())
            elif isinstance(output, (tuple, list)):
                values = list(output)
            else:
                values = [output]
            values = [value for value in values if hasattr(value, "shape")]
            if len(values) < self.count:
                raise RuntimeError(f"partition returned {len(values)} tensor outputs, expected at least {self.count}")
            return tuple(values[: self.count])

    return _Wrapper(module, output_count)


def import_factory(factory: str) -> Any:
    module_name, sep, attr = factory.partition(":")
    if not sep:
        raise SystemExit("--model-factory must be module:function")
    module = importlib.import_module(module_name)
    target = module
    for name in attr.split("."):
        target = getattr(target, name)
    return target


def resolve_attr(root: Any, path: str) -> Any | None:
    target = root
    for name in path.split("."):
        if not hasattr(target, name):
            return None
        target = getattr(target, name)
    return target


def resolve_partition_module(model: Any, candidates: list[str]) -> Any | None:
    for candidate in candidates:
        module = resolve_attr(model, candidate)
        if module is not None:
            return module
    return None


def load_model(args: argparse.Namespace) -> Any:
    factory = import_factory(args.model_factory)
    kwargs = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "model_config": str(args.model_config) if args.model_config else None,
        "device": args.device,
    }
    try:
        model = factory(**kwargs)
    except TypeError:
        model = factory(args.checkpoint, args.model_config, args.device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def torch_dtype(torch: Any, dtype: str) -> Any:
    return getattr(torch, DTYPES[dtype])


def dummy_inputs(torch: Any, inputs: list[dict[str, Any]], device: str) -> tuple[Any, ...]:
    tensors = []
    for item in inputs:
        dtype = torch_dtype(torch, str(item["dtype"]))
        shape = list(item["concrete_shape"])
        if str(item["dtype"]).startswith("float"):
            tensor = torch.zeros(shape, dtype=dtype, device=device)
        else:
            tensor = torch.zeros(shape, dtype=dtype, device=device)
        tensors.append(tensor)
    return tuple(tensors)


def export_partition(torch: Any, model: Any, contract_path: Path, args: argparse.Namespace) -> None:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    module = resolve_partition_module(model, contract["upstream_candidates"])
    if module is None:
        candidates = ", ".join(contract["upstream_candidates"])
        raise SystemExit(f"failed to resolve partition {contract['name']} from candidates: {candidates}")
    if hasattr(module, "eval"):
        module.eval()

    inputs = dummy_inputs(torch, contract["inputs"], args.device)
    input_names = [item["name"] for item in contract["inputs"]]
    output_names = [item["name"] for item in contract["outputs"]]
    onnx_path = Path(contract["onnx_path"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = make_contract_module(torch, module, len(output_names))
    torch.onnx.export(
        wrapper,
        inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"exported {contract['name']}: {onnx_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export SAM3/SAM3.1 partitions to ONNX using an authorized upstream model factory.")
    parser.add_argument("--model-factory", required=True, help="Python callable as module:function; returns an upstream SAM3 model")
    parser.add_argument("--checkpoint", type=Path, help="authorized upstream SAM3/SAM3.1 checkpoint")
    parser.add_argument("--model-config", type=Path, help="optional upstream model config")
    parser.add_argument("--sam3-repo", type=Path, help="local upstream SAM3 repository to prepend to PYTHONPATH")
    parser.add_argument("--out-dir", type=Path, default=Path("build/sam3_export"))
    parser.add_argument("--partition", action="append", choices=sorted(PARTITIONS), help="partition to export; repeatable")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dry-run", action="store_true", help="generate contracts and resolve modules without writing ONNX")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--embed-channels", type=int, default=256)
    parser.add_argument("--image-stride", type=int, default=16)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--max-geometry-tokens", type=int, default=256)
    parser.add_argument("--max-objects", type=int, default=256)
    parser.add_argument("--max-memory-tokens", type=int, default=256)
    parser.add_argument("--mask-size", type=int, default=256)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.sam3_repo:
        sys.path.insert(0, str(args.sam3_repo))

    partition_names = args.partition or list(PARTITIONS)
    partitions = selected_partitions(partition_names)
    write_contracts(partitions, args.out_dir, args.checkpoint, args.sam3_repo, export_dims(args))
    model = load_model(args)

    torch = None
    if not args.dry_run:
        try:
            import torch as torch_module  # type: ignore
        except ImportError as error:
            raise SystemExit("ONNX export requires PyTorch; install torch in the export environment") from error
        torch = torch_module

    for spec in partitions:
        contract_path = args.out_dir / "contracts" / f"{spec.name}.json"
        if args.dry_run:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            module = resolve_partition_module(model, contract["upstream_candidates"])
            status = "resolved" if module is not None else "missing"
            print(f"{spec.name}: {status}")
            continue
        export_partition(torch, model, contract_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
