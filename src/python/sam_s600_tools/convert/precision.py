#!/usr/bin/env python3
"""Convert ONNX floating tensors between fp32/fp16/bf16 for HBDK experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import GraphProto, TensorProto, helper, numpy_helper

FLOAT_TYPES = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16}
TARGET_TYPES = {
    "fp32": TensorProto.FLOAT,
    "fp16": TensorProto.FLOAT16,
    "bf16": TensorProto.BFLOAT16,
}


def to_bfloat16_bits(array: np.ndarray) -> np.ndarray:
    values = array.astype(np.float32, copy=False).view(np.uint32)
    rounded = values + np.uint32(0x00008000)
    return (rounded >> np.uint32(16)).astype(np.uint16)


def make_bfloat16_initializer(tensor: TensorProto) -> TensorProto:
    array = numpy_helper.to_array(tensor).astype(np.float32, copy=False)
    converted = TensorProto()
    converted.name = tensor.name
    converted.data_type = TensorProto.BFLOAT16
    converted.dims.extend(tensor.dims)
    converted.raw_data = to_bfloat16_bits(array).tobytes()
    return converted


def convert_initializer(tensor: TensorProto, target: int) -> TensorProto:
    if tensor.data_type not in FLOAT_TYPES or tensor.data_type == target:
        return tensor
    if target == TensorProto.FLOAT:
        array = numpy_helper.to_array(tensor).astype(np.float32)
        return numpy_helper.from_array(array, tensor.name)
    if target == TensorProto.FLOAT16:
        array = numpy_helper.to_array(tensor).astype(np.float16)
        return numpy_helper.from_array(array, tensor.name)
    if target == TensorProto.BFLOAT16:
        return make_bfloat16_initializer(tensor)
    raise ValueError(f"unsupported target dtype: {target}")


def convert_value_info(value: onnx.ValueInfoProto, target: int) -> None:
    tensor_type = value.type.tensor_type
    if tensor_type.elem_type in FLOAT_TYPES:
        tensor_type.elem_type = target


def iter_subgraphs(node: onnx.NodeProto) -> list[GraphProto]:
    graphs: list[GraphProto] = []
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.GRAPH:
            graphs.append(attr.g)
        elif attr.type == onnx.AttributeProto.GRAPHS:
            graphs.extend(attr.graphs)
    return graphs


def convert_cast_nodes(graph: GraphProto, target: int) -> None:
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i in FLOAT_TYPES:
                    attr.i = target
        for subgraph in iter_subgraphs(node):
            convert_cast_nodes(subgraph, target)


def convert_constants(graph: GraphProto, target: int) -> None:
    for node in graph.node:
        if node.op_type != "Constant":
            for subgraph in iter_subgraphs(node):
                convert_constants(subgraph, target)
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.t.data_type in FLOAT_TYPES:
                attr.t.CopyFrom(convert_initializer(attr.t, target))
            elif attr.name == "value_float" and target != TensorProto.FLOAT:
                value = np.array([attr.f], dtype=np.float32)
                if target == TensorProto.FLOAT16:
                    tensor = numpy_helper.from_array(value.astype(np.float16), node.output[0] if node.output else "")
                    attr.name = "value"
                    attr.type = onnx.AttributeProto.TENSOR
                    attr.t.CopyFrom(tensor)
                elif target == TensorProto.BFLOAT16:
                    tensor = TensorProto()
                    tensor.name = node.output[0] if node.output else ""
                    tensor.data_type = TensorProto.BFLOAT16
                    tensor.dims.extend([1])
                    tensor.raw_data = to_bfloat16_bits(value).tobytes()
                    attr.name = "value"
                    attr.type = onnx.AttributeProto.TENSOR
                    attr.t.CopyFrom(tensor)
            elif attr.name == "value_floats" and target != TensorProto.FLOAT:
                value = np.array(list(attr.floats), dtype=np.float32)
                if target == TensorProto.FLOAT16:
                    tensor = numpy_helper.from_array(value.astype(np.float16), node.output[0] if node.output else "")
                else:
                    tensor = TensorProto()
                    tensor.name = node.output[0] if node.output else ""
                    tensor.data_type = TensorProto.BFLOAT16
                    tensor.dims.extend([len(value)])
                    tensor.raw_data = to_bfloat16_bits(value).tobytes()
                attr.name = "value"
                attr.type = onnx.AttributeProto.TENSOR
                attr.t.CopyFrom(tensor)
        for subgraph in iter_subgraphs(node):
            convert_constants(subgraph, target)


def convert_graph(graph: GraphProto, target: int, keep_io_fp32: bool) -> None:
    for index, tensor in enumerate(graph.initializer):
        converted = convert_initializer(tensor, target)
        if converted is not tensor:
            graph.initializer[index].CopyFrom(converted)

    convert_constants(graph, target)
    convert_cast_nodes(graph, target)

    value_infos = list(graph.value_info)
    if not keep_io_fp32:
        value_infos += list(graph.input) + list(graph.output)
    for value in value_infos:
        convert_value_info(value, target)

    for node in graph.node:
        for subgraph in iter_subgraphs(node):
            convert_graph(subgraph, target, keep_io_fp32)


def convert_initializers_only(graph: GraphProto, target: int) -> None:
    for index, tensor in enumerate(graph.initializer):
        converted = convert_initializer(tensor, target)
        if converted is not tensor:
            graph.initializer[index].CopyFrom(converted)
    for node in graph.node:
        for subgraph in iter_subgraphs(node):
            convert_initializers_only(subgraph, target)


def cast_initializer_users_to_fp32(graph: GraphProto) -> None:
    initializer_names = {tensor.name for tensor in graph.initializer if tensor.data_type == TensorProto.FLOAT16}
    if not initializer_names:
        return

    new_nodes = []
    casted: set[str] = set()
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name not in initializer_names:
                continue
            cast_name = name + "__fp16_to_fp32"
            node.input[index] = cast_name
            if name in casted:
                continue
            casted.add(name)
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[name],
                    outputs=[cast_name],
                    name=cast_name,
                    to=TensorProto.FLOAT,
                )
            )
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)

    for node in graph.node:
        for subgraph in iter_subgraphs(node):
            cast_initializer_users_to_fp32(subgraph)


def convert_model(input_path: Path, output_path: Path, precision: str, keep_io_fp32: bool, weights_only: bool, cast_weights_to_fp32: bool) -> None:
    target = TARGET_TYPES[precision]
    model = onnx.load(input_path, load_external_data=True)

    if weights_only:
        convert_initializers_only(model.graph, target)
        if cast_weights_to_fp32:
            cast_initializer_users_to_fp32(model.graph)
    else:
        convert_graph(model.graph, target, keep_io_fp32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.name + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.checker.check_model(output_path)
    print(f"wrote {precision} ONNX: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert ONNX floating tensors to fp16/bf16/fp32.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--precision", choices=sorted(TARGET_TYPES), default="fp16")
    parser.add_argument("--keep-io-fp32", action="store_true", help="convert internals/weights but keep graph inputs and outputs fp32")
    parser.add_argument("--weights-only", action="store_true", help="convert only floating initializers; keep graph value/input/output types unchanged")
    parser.add_argument("--cast-weights-to-fp32", action="store_true", help="after --weights-only, insert Cast nodes so fp16/bf16 initializers feed fp32 graph values")
    args = parser.parse_args()
    convert_model(args.input, args.output, args.precision, args.keep_io_fp32, args.weights_only, args.cast_weights_to_fp32)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
