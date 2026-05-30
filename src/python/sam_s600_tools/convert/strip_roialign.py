#!/usr/bin/env python3
"""Prepare SAM3 ONNX graphs for HBDK conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper

DTYPE_TO_CONTRACT = {
    TensorProto.FLOAT: "float32",
    TensorProto.FLOAT16: "float16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.UINT8: "uint8",
    TensorProto.BOOL: "bool",
}
CONTRACT_TO_DTYPE = {value: key for key, value in DTYPE_TO_CONTRACT.items()}


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str]:
    return [dim.dim_value if dim.dim_value else dim.dim_param for dim in value.type.tensor_type.shape.dim]


def tensor_dtype(value: onnx.ValueInfoProto) -> int:
    return value.type.tensor_type.elem_type


def graph_values(model: onnx.ModelProto, include_inferred: bool = False) -> dict[str, onnx.ValueInfoProto]:
    values: dict[str, onnx.ValueInfoProto] = {}
    source = model
    if include_inferred:
        try:
            source = onnx.shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
        except Exception:
            source = model
    for value in list(source.graph.input) + list(source.graph.value_info) + list(source.graph.output):
        values[value.name] = value
    return values


def parse_shape(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(part) for part in text.replace("x", ",").split(",") if part]


def parse_named_shapes(items: list[str]) -> dict[str, tuple[list[int] | None, int | None]]:
    shapes: dict[str, tuple[list[int] | None, int | None]] = {}
    for item in items:
        parts = item.split(":")
        if len(parts) == 1:
            shapes[item] = (None, None)
        elif len(parts) == 2:
            shapes[parts[0]] = (parse_shape(parts[1]), None)
        elif len(parts) == 3:
            dtype = CONTRACT_TO_DTYPE.get(parts[2])
            if dtype is None:
                raise SystemExit(f"unsupported cut input dtype {parts[2]} for {parts[0]}")
            shapes[parts[0]] = (parse_shape(parts[1]), dtype)
        else:
            raise SystemExit(f"invalid --cut-input format: {item}")
    return shapes

def contract_by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in items}


def concrete_shape(item: dict[str, Any] | None) -> list[int] | None:
    if not item:
        return None
    shape = item.get("concrete_shape") or item.get("shape")
    if shape and all(isinstance(dim, int) for dim in shape):
        return list(shape)
    return None


def output_contract_shape(contract: dict[str, Any], output_name: str) -> list[int] | None:
    return concrete_shape(contract_by_name(contract.get("outputs", [])).get(output_name))


def tensor_contract(value: onnx.ValueInfoProto, source: str | None = None, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    shape = tensor_shape(value)
    fallback_shape = concrete_shape(fallback)
    if all(isinstance(dim, int) for dim in shape):
        concrete = list(shape)
    elif fallback_shape is not None:
        concrete = fallback_shape
    else:
        raise SystemExit(f"tensor {value.name} has non-concrete shape {shape}")
    dtype = DTYPE_TO_CONTRACT.get(tensor_dtype(value))
    if dtype is None and fallback and isinstance(fallback.get("dtype"), str):
        dtype = str(fallback["dtype"])
    if dtype is None:
        raise SystemExit(f"tensor {value.name} has unsupported dtype {tensor_dtype(value)}")
    symbolic = fallback.get("symbolic_shape") or fallback.get("shape") if fallback else None
    return {
        "name": value.name,
        "dtype": dtype,
        "symbolic_shape": symbolic or shape,
        "concrete_shape": concrete,
        "source": source or str(fallback.get("source", "runtime") if fallback else "runtime"),
    }


def attr(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for item in node.attribute:
        if item.name == name:
            return helper.get_attribute_value(item)
    return default


def infer_roi_shape(model: onnx.ModelProto, roi: onnx.NodeProto, fallback_shape: list[int] | None) -> list[int]:
    if fallback_shape:
        return fallback_shape
    values = graph_values(model)
    feature_shape = tensor_shape(values[roi.input[0]]) if roi.input[0] in values else []
    rois_shape = tensor_shape(values[roi.input[1]]) if roi.input[1] in values else []
    channels = feature_shape[1] if len(feature_shape) >= 2 and isinstance(feature_shape[1], int) else 256
    num_rois = rois_shape[0] if rois_shape and isinstance(rois_shape[0], int) else None
    if num_rois is None:
        for value in model.graph.input:
            shape = tensor_shape(value)
            if len(shape) == 3 and shape[-1] == 4 and isinstance(shape[1], int):
                num_rois = shape[1]
                break
    if num_rois is None:
        raise SystemExit(f"failed to infer RoiAlign ROI count for node {roi.name}; pass --roi-shape")
    return [int(num_rois), int(channels), int(attr(roi, "output_height")), int(attr(roi, "output_width"))]


def required_node_indices(model: onnx.ModelProto) -> set[int]:
    required_values = {output.name for output in model.graph.output}
    keep: set[int] = set()
    for index in range(len(model.graph.node) - 1, -1, -1):
        node = model.graph.node[index]
        if any(output in required_values for output in node.output):
            keep.add(index)
            required_values.update(node_dependencies(node))
    return keep


def prune_graph(model: onnx.ModelProto) -> None:
    keep = required_node_indices(model)
    nodes = [node for index, node in enumerate(model.graph.node) if index in keep]
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    used = {input_name for node in model.graph.node for input_name in node.input if input_name}
    outputs = {output.name for output in model.graph.output}
    inputs = [value for value in model.graph.input if value.name in used or value.name in outputs]
    del model.graph.input[:]
    model.graph.input.extend(inputs)


def replace_roialign(model: onnx.ModelProto, input_name: str, roi_shape: list[int] | None) -> list[tuple[str, list[int]]]:
    roi_nodes = [node for node in model.graph.node if node.op_type == "RoiAlign"]
    added: list[tuple[str, list[int]]] = []
    existing_inputs = {value.name for value in model.graph.input}
    for index, roi in enumerate(roi_nodes):
        old_output = roi.output[0]
        new_input = input_name if len(roi_nodes) == 1 else f"{input_name}_{index}"
        if new_input in existing_inputs:
            raise SystemExit(f"graph already has input named {new_input}")
        shape = infer_roi_shape(model, roi, roi_shape)
        model.graph.input.append(helper.make_tensor_value_info(new_input, TensorProto.FLOAT, shape))
        existing_inputs.add(new_input)
        added.append((new_input, shape))
        for node in model.graph.node:
            for input_index, value in enumerate(node.input):
                if value == old_output:
                    node.input[input_index] = new_input
    if roi_nodes:
        nodes = [node for node in model.graph.node if node.op_type != "RoiAlign"]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        prune_graph(model)
    return added


def shape_for_einsum(model: onnx.ModelProto, node: onnx.NodeProto, contract: dict[str, Any]) -> tuple[int, int, int, int, int]:
    values = graph_values(model)
    a_shape = tensor_shape(values[node.input[0]]) if node.input[0] in values else []
    b_shape = tensor_shape(values[node.input[1]]) if node.input[1] in values else []
    out_shape = tensor_shape(values[node.output[0]]) if node.output[0] in values else []
    contract_shape = output_contract_shape(contract, node.output[0])
    shape = contract_shape or out_shape
    if len(b_shape) == 4 and all(isinstance(dim, int) for dim in b_shape):
        batch, channels, height, width = (int(dim) for dim in b_shape)
    elif len(shape) == 4 and all(isinstance(shape[i], int) for i in (0, 2, 3)):
        batch, height, width = int(shape[0]), int(shape[2]), int(shape[3])
        channels = int(a_shape[2]) if len(a_shape) == 3 and isinstance(a_shape[2], int) else 256
    else:
        raise SystemExit(f"failed to infer shape for Einsum {node.name}")

    queries = None
    if len(a_shape) == 3 and isinstance(a_shape[1], int):
        queries = int(a_shape[1])
    elif len(shape) == 4 and isinstance(shape[1], int):
        queries = int(shape[1])
    if queries is None:
        raise SystemExit(f"failed to infer query count for Einsum {node.name}; update contract output shape")
    return batch, queries, channels, height, width


def replace_einsum(model: onnx.ModelProto, contract: dict[str, Any]) -> int:
    new_nodes = []
    replaced = 0
    for node in model.graph.node:
        equation = attr(node, "equation")
        if node.op_type != "Einsum" or equation not in (b"bqc,bchw->bqhw", "bqc,bchw->bqhw"):
            new_nodes.append(node)
            continue
        batch, queries, channels, height, width = shape_for_einsum(model, node, contract)
        prefix = (node.name or node.output[0]).replace("/", "_")
        b_flat_shape = f"{prefix}_b_flat_shape"
        out_shape = f"{prefix}_out_shape"
        b_flat = f"{prefix}_b_flat"
        matmul = f"{prefix}_matmul"
        model.graph.initializer.append(helper.make_tensor(b_flat_shape, TensorProto.INT64, [3], [batch, channels, height * width]))
        model.graph.initializer.append(helper.make_tensor(out_shape, TensorProto.INT64, [4], [batch, queries, height, width]))
        new_nodes.extend(
            [
                helper.make_node("Reshape", [node.input[1], b_flat_shape], [b_flat], name=f"{prefix}_ReshapeB"),
                helper.make_node("MatMul", [node.input[0], b_flat], [matmul], name=f"{prefix}_MatMul"),
                helper.make_node("Reshape", [matmul, out_shape], list(node.output), name=f"{prefix}_ReshapeOut"),
            ]
        )
        replaced += 1
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return replaced


def make_graph_input(name: str, value: onnx.ValueInfoProto | None, shape_override: list[int] | None = None, dtype_override: int | None = None) -> onnx.ValueInfoProto:
    shape = shape_override or (tensor_shape(value) if value is not None else None)
    if shape is None or not all(isinstance(dim, int) for dim in shape):
        raise SystemExit(f"cut input {name} has non-concrete shape {shape}")
    dtype = dtype_override or (tensor_dtype(value) if value is not None else TensorProto.FLOAT)
    return helper.make_tensor_value_info(name, dtype, list(shape))


def cut_graph_inputs(model: onnx.ModelProto, cut_specs: dict[str, tuple[list[int] | None, int | None]]) -> None:
    if not cut_specs:
        return
    cut_names = set(cut_specs)
    values = graph_values(model, include_inferred=True)
    missing = [name for name, (shape, dtype) in cut_specs.items() if name not in values and (shape is None or dtype is None)]
    if missing:
        raise SystemExit(f"cut inputs missing value_info: {missing}")
    existing_inputs = {value.name for value in model.graph.input}
    for name, (shape_override, dtype_override) in cut_specs.items():
        if name not in existing_inputs:
            model.graph.input.append(make_graph_input(name, values.get(name), shape_override, dtype_override))
            existing_inputs.add(name)
    producer_indices = {
        index
        for index, node in enumerate(model.graph.node)
        if any(output in cut_names for output in node.output)
    }
    nodes = [node for index, node in enumerate(model.graph.node) if index not in producer_indices]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    prune_graph(model)


def select_outputs(model: onnx.ModelProto, output_specs: dict[str, tuple[list[int] | None, int | None]]) -> None:
    if not output_specs:
        return
    values = graph_values(model)
    outputs = []
    for name, (shape_override, dtype_override) in output_specs.items():
        value = values.get(name)
        if shape_override is not None or dtype_override is not None:
            outputs.append(make_graph_input(name, value, shape_override, dtype_override))
        elif value is not None:
            outputs.append(value)
        else:
            raise SystemExit(f"selected output missing value_info: {name}")
    del model.graph.output[:]
    model.graph.output.extend(outputs)


def update_contract(contract: dict[str, Any], model: onnx.ModelProto, onnx_path: Path, out_path: Path) -> None:
    contract["onnx_path"] = str(onnx_path)
    original_inputs = contract_by_name(contract.get("inputs", []))
    original_outputs = contract_by_name(contract.get("outputs", []))
    contract["inputs"] = [tensor_contract(value, fallback=original_inputs.get(value.name)) for value in model.graph.input]
    outputs = []
    for value in model.graph.output:
        item = tensor_contract(value, source=contract.get("name", "runtime"), fallback=original_outputs.get(value.name))
        outputs.append(item)
    contract["outputs"] = outputs
    contract.pop("run_on_cpu", None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")


def external_graph_inputs(graph: onnx.GraphProto) -> set[str]:
    local_values = {value.name for value in graph.input}
    local_values.update(initializer.name for initializer in graph.initializer)
    produced = {output for node in graph.node for output in node.output if output}
    external: set[str] = set()
    for node in graph.node:
        for input_name in node.input:
            if input_name and input_name not in local_values and input_name not in produced:
                external.add(input_name)
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                for input_name in external_graph_inputs(attribute.g):
                    if input_name not in local_values and input_name not in produced:
                        external.add(input_name)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    for input_name in external_graph_inputs(subgraph):
                        if input_name not in local_values and input_name not in produced:
                            external.add(input_name)
    return external


def node_dependencies(node: onnx.NodeProto) -> set[str]:
    dependencies = {input_name for input_name in node.input if input_name}
    for attribute in node.attribute:
        if attribute.type == onnx.AttributeProto.GRAPH:
            dependencies.update(external_graph_inputs(attribute.g))
        elif attribute.type == onnx.AttributeProto.GRAPHS:
            for subgraph in attribute.graphs:
                dependencies.update(external_graph_inputs(subgraph))
    return dependencies


def topo_sort_graph(graph: onnx.GraphProto) -> None:
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                topo_sort_graph(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    topo_sort_graph(subgraph)

    produced = {output for node in graph.node for output in node.output if output}
    available = {value.name for value in graph.input}
    available.update(initializer.name for initializer in graph.initializer)
    pending = list(graph.node)
    ordered = []
    while pending:
        progressed = False
        next_pending = []
        for node in pending:
            if all(input_name in available or input_name not in produced for input_name in node_dependencies(node)):
                ordered.append(node)
                available.update(output for output in node.output if output)
                progressed = True
            else:
                next_pending.append(node)
        if not progressed:
            break
        pending = next_pending
    ordered.extend(pending)
    del graph.node[:]
    graph.node.extend(ordered)


def topo_sort_nodes(model: onnx.ModelProto) -> None:
    topo_sort_graph(model.graph)


def save_model(model: onnx.ModelProto, output: Path) -> None:
    topo_sort_nodes(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(output),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output.name}.data",
        size_threshold=1024,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare SAM3 ONNX graphs for HBDK.")
    parser.add_argument("input_onnx", type=Path)
    parser.add_argument("output_onnx", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out-contract", type=Path, required=True)
    parser.add_argument("--replace-roialign", action="store_true")
    parser.add_argument("--roi-input-name", default="roi_features")
    parser.add_argument("--roi-shape")
    parser.add_argument("--replace-einsum", action="store_true")
    parser.add_argument("--cut-input", action="append", default=[], help="Treat an internal tensor as an external graph input, optionally name:shape:dtype; repeatable")
    parser.add_argument("--output", action="append", default=[], help="Keep only this graph output, optionally name:shape:dtype; repeatable")
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    model = onnx.load(str(args.input_onnx), load_external_data=True)
    if args.output:
        output_specs = parse_named_shapes(args.output)
        select_outputs(model, output_specs)
        print(f"selected outputs: {len(output_specs)}")
    if args.replace_roialign:
        for name, shape in replace_roialign(model, args.roi_input_name, parse_shape(args.roi_shape)):
            print(f"added input {name}: {shape}")
    if args.replace_einsum:
        print(f"replaced einsum nodes: {replace_einsum(model, contract)}")
    if args.cut_input:
        cut_shapes = parse_named_shapes(args.cut_input)
        cut_graph_inputs(model, cut_shapes)
        print(f"cut graph inputs: {len(cut_shapes)}")
    prune_graph(model)
    save_model(model, args.output_onnx)
    onnx.checker.check_model(str(args.output_onnx))
    update_contract(contract, model, args.output_onnx, args.out_contract)
    print(f"wrote {args.output_onnx}")
    print(f"wrote {args.out_contract}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
