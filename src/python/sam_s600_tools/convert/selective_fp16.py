#!/usr/bin/env python3
"""Apply FP16 to selected SAM3 ONNX subgraph regions while keeping the rest FP32.

This is for S600 experiments where full-graph FP16 fails in the text softmax/clip
path.  The converter rewrites only nodes selected by name prefix (by default the
mask decoder tail), inserts FP32->FP16 casts at selected-region boundaries, casts
selected floating weights/constants to FP16, and optionally leaves selected graph
outputs as FP16.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import AttributeProto, GraphProto, TensorProto, helper, numpy_helper

FLOAT_TYPES = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16}

DEFAULT_PREFIXES = (
    "/wrapped/mask_decoder/pixel_decoder/conv_layers.1/",
    "/wrapped/mask_decoder/pixel_decoder/norms.1/",
    "/wrapped/mask_decoder/pixel_decoder/Relu_1",
    "/wrapped/mask_decoder/instance_projection/",
    "/wrapped/mask_decoder/semantic_projection/",
    "/wrapped/mask_decoder/mask_embedder/",
    "_wrapped_mask_decoder_Einsum_",
)

DEFAULT_FP16_OUTPUTS = ("mask_logits", "semantic_segmentation")


def _tensor_elem_type(value: onnx.ValueInfoProto) -> int | None:
    tensor_type = value.type.tensor_type
    if tensor_type.elem_type:
        return tensor_type.elem_type
    return None


def _collect_elem_types(graph: GraphProto) -> dict[str, int]:
    types: dict[str, int] = {}
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        elem_type = _tensor_elem_type(value)
        if elem_type is not None:
            types[value.name] = elem_type
    for tensor in graph.initializer:
        types[tensor.name] = tensor.data_type
    return types


def _collect_inferred_elem_types(model: onnx.ModelProto) -> dict[str, int]:
    """Best-effort ONNX shape/type inference for boundary dtype decisions."""
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        return {}
    return _collect_elem_types(inferred.graph)


def _convert_initializer_fp16(tensor: TensorProto, name: str | None = None) -> TensorProto:
    if tensor.data_type == TensorProto.FLOAT16:
        out = TensorProto()
        out.CopyFrom(tensor)
        if name is not None:
            out.name = name
        return out
    if tensor.data_type != TensorProto.FLOAT:
        out = TensorProto()
        out.CopyFrom(tensor)
        if name is not None:
            out.name = name
        return out
    arr = numpy_helper.to_array(tensor).astype(np.float16)
    return numpy_helper.from_array(arr, name or tensor.name)


def _convert_constant_attr_fp16(attr: onnx.AttributeProto, output_name: str) -> None:
    if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
        attr.t.CopyFrom(_convert_initializer_fp16(attr.t, attr.t.name or output_name))
    elif attr.name == "value_float":
        arr = np.array([attr.f], dtype=np.float16)
        tensor = numpy_helper.from_array(arr, output_name)
        attr.name = "value"
        attr.type = AttributeProto.TENSOR
        attr.t.CopyFrom(tensor)
    elif attr.name == "value_floats":
        arr = np.array(list(attr.floats), dtype=np.float16)
        tensor = numpy_helper.from_array(arr, output_name)
        attr.name = "value"
        attr.type = AttributeProto.TENSOR
        attr.t.CopyFrom(tensor)


def _set_value_elem_type(values: list[onnx.ValueInfoProto], name: str, elem_type: int) -> bool:
    for value in values:
        if value.name == name and value.type.HasField("tensor_type"):
            if value.type.tensor_type.elem_type in FLOAT_TYPES:
                value.type.tensor_type.elem_type = elem_type
                return True
    return False


def _unique(base: str, used: set[str]) -> str:
    if base not in used:
        used.add(base)
        return base
    i = 1
    while f"{base}_{i}" in used:
        i += 1
    name = f"{base}_{i}"
    used.add(name)
    return name


def _selected(node: onnx.NodeProto, prefixes: tuple[str, ...]) -> bool:
    return any(node.name.startswith(prefix) for prefix in prefixes)


def _cast_node_name(tensor_name: str, suffix: str, used: set[str]) -> str:
    safe = tensor_name.replace("/", "_").replace(":", "_").replace(".", "_")
    return _unique(f"Cast_{safe}_{suffix}", used)


def convert_model(input_path: Path, output_path: Path, prefixes: tuple[str, ...], fp16_outputs: set[str]) -> None:
    model = onnx.load(input_path, load_external_data=True)
    inferred_elem_types = _collect_inferred_elem_types(model)
    graph = model.graph

    selected_indices = {i for i, node in enumerate(graph.node) if _selected(node, prefixes)}
    if not selected_indices:
        raise SystemExit("no nodes matched --prefix; refusing to write unchanged model")

    producer: dict[str, int] = {}
    for i, node in enumerate(graph.node):
        for out in node.output:
            producer[out] = i

    consumers: dict[str, list[int]] = defaultdict(list)
    for i, node in enumerate(graph.node):
        for inp in node.input:
            if inp:
                consumers[inp].append(i)

    elem_types = _collect_elem_types(graph)
    # Existing exported ONNX may not carry complete value_info. Use inferred types
    # to decide which region-boundary tensors are FP32 and need casts.
    elem_types.update({k: v for k, v in inferred_elem_types.items() if k not in elem_types})

    initializer_by_name = {tensor.name: tensor for tensor in graph.initializer}
    selected_outputs = {out for i in selected_indices for out in graph.node[i].output}
    graph_output_names = {out.name for out in graph.output}
    used_names = {v.name for v in list(graph.input) + list(graph.value_info) + list(graph.output)}
    used_names.update(initializer_by_name)
    for node in graph.node:
        used_names.update(x for x in node.input if x)
        used_names.update(x for x in node.output if x)
        if node.name:
            used_names.add(node.name)

    # Convert or clone floating initializers used by selected nodes. If an
    # initializer is also used outside the selected region, keep the FP32 original
    # and redirect selected users to a FP16 clone.
    new_initializers: list[TensorProto] = []
    for init_name, tensor in list(initializer_by_name.items()):
        if tensor.data_type != TensorProto.FLOAT:
            continue
        users = consumers.get(init_name, [])
        if not users or not any(i in selected_indices for i in users):
            continue
        if all(i in selected_indices for i in users):
            idx = next(i for i, t in enumerate(graph.initializer) if t.name == init_name)
            graph.initializer[idx].CopyFrom(_convert_initializer_fp16(tensor))
            elem_types[init_name] = TensorProto.FLOAT16
        else:
            clone_name = _unique(init_name + "__fp16", used_names)
            new_initializers.append(_convert_initializer_fp16(tensor, clone_name))
            elem_types[clone_name] = TensorProto.FLOAT16
            for i in users:
                if i not in selected_indices:
                    continue
                node = graph.node[i]
                for j, inp in enumerate(node.input):
                    if inp == init_name:
                        node.input[j] = clone_name
    graph.initializer.extend(new_initializers)
    initializer_by_name = {tensor.name: tensor for tensor in graph.initializer}

    new_nodes: list[onnx.NodeProto] = []
    boundary_casts: dict[str, str] = {}
    output_back_casts: dict[str, str] = {}
    inserted_to_fp16 = 0
    inserted_to_fp32 = 0

    for i, node in enumerate(graph.node):
        if i not in selected_indices:
            # Rewire non-selected consumers of selected FP16 outputs back to FP32.
            for j, inp in enumerate(node.input):
                if inp in output_back_casts:
                    node.input[j] = output_back_casts[inp]
            new_nodes.append(node)
            continue

        # Insert/reuse FP32->FP16 boundary casts for floating non-initializer inputs
        # that come from outside the selected region.
        for j, inp in enumerate(node.input):
            if not inp or inp in initializer_by_name:
                continue
            input_type = elem_types.get(inp)
            # Some exported ONNX graphs do not retain value_info for all
            # intermediates. Known non-floating tensors (shape/index/bool) must not
            # be cast, but unknown boundary tensors in this selected compute tail
            # are assumed floating and cast to FP16.
            if input_type in {TensorProto.FLOAT16, TensorProto.BFLOAT16}:
                continue
            if input_type is not None and input_type != TensorProto.FLOAT:
                continue
            prod_idx = producer.get(inp)
            if prod_idx in selected_indices:
                continue
            cast_out = boundary_casts.get(inp)
            if cast_out is None:
                cast_out = _unique(inp + "__to_fp16", used_names)
                cast_node = helper.make_node(
                    "Cast",
                    inputs=[inp],
                    outputs=[cast_out],
                    name=_cast_node_name(inp, "to_fp16", used_names),
                    to=TensorProto.FLOAT16,
                )
                boundary_casts[inp] = cast_out
                elem_types[cast_out] = TensorProto.FLOAT16
                new_nodes.append(cast_node)
                inserted_to_fp16 += 1
            node.input[j] = cast_out

        # FP32 Casts inside selected region should become FP16 Casts; leave int/bool
        # casts untouched.
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
        elif node.op_type == "Constant" and node.output:
            for attr in node.attribute:
                _convert_constant_attr_fp16(attr, node.output[0])

        new_nodes.append(node)

        for out in node.output:
            if elem_types.get(out) in (TensorProto.FLOAT, TensorProto.FLOAT16, None):
                elem_types[out] = TensorProto.FLOAT16
            external_users = [u for u in consumers.get(out, []) if u not in selected_indices]
            needs_fp32_graph_output = out in graph_output_names and out not in fp16_outputs
            if external_users or needs_fp32_graph_output:
                cast_back = _unique(out + "__to_fp32", used_names)
                output_back_casts[out] = cast_back
                new_nodes.append(
                    helper.make_node(
                        "Cast",
                        inputs=[out],
                        outputs=[cast_back],
                        name=_cast_node_name(out, "to_fp32", used_names),
                        to=TensorProto.FLOAT,
                    )
                )
                elem_types[cast_back] = TensorProto.FLOAT
                inserted_to_fp32 += 1

    del graph.node[:]
    graph.node.extend(new_nodes)

    all_value_infos = list(graph.value_info) + list(graph.input) + list(graph.output)
    for name in selected_outputs | set(boundary_casts.values()):
        _set_value_elem_type(all_value_infos, name, TensorProto.FLOAT16)
    for out in graph.output:
        if out.name in fp16_outputs and out.type.tensor_type.elem_type in FLOAT_TYPES:
            out.type.tensor_type.elem_type = TensorProto.FLOAT16
        elif out.name in output_back_casts:
            out.type.tensor_type.elem_type = TensorProto.FLOAT

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
    print(
        f"wrote selective-fp16 ONNX: {output_path} "
        f"nodes={len(selected_indices)} fp16_initializers={sum(1 for t in graph.initializer if t.data_type == TensorProto.FLOAT16)} "
        f"casts_to_fp16={inserted_to_fp16} casts_to_fp32={inserted_to_fp32} "
        f"fp16_outputs={','.join(sorted(fp16_outputs))}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="node-name prefix to convert; repeatable (default: mask decoder tail)",
    )
    parser.add_argument(
        "--fp16-outputs",
        default=",".join(DEFAULT_FP16_OUTPUTS),
        help="comma-separated graph outputs to leave as FP16 (default: mask_logits,semantic_segmentation)",
    )
    args = parser.parse_args()
    prefixes = tuple(args.prefix or DEFAULT_PREFIXES)
    fp16_outputs = {name.strip() for name in args.fp16_outputs.split(",") if name.strip()}
    convert_model(args.input, args.output, prefixes, fp16_outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
