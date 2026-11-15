#!/usr/bin/env python3
"""Writes the single-file model (.tym, docs/ModelFormat.md) from a YOLO export folder.

usage: yolo_make_tym.py <yolo11|yolo26> <export folder> <output.tym> [--per-tensor]

The export folder holds <name>_graph.json (the operation list and the class names), <name>_graph_weights.bin
(float32 parameters), <name>_int8_weights.bin and <name>_int8_scales.bin. The file layout is the one of src/TinyYolo/app/ModelFileWriter.h (the desktop tests read it back).
"""
import json
import os
import struct
import sys
import zlib

INPUT_SIZE = 640  # the input the exported graphs are built for (kModel in the export tables)
KINDS = {"Conv": 0, "Add": 1, "Chunk": 2, "Concat": 3, "MaxPool": 4, "Upsample": 5, "Attention": 6}
SECTION = {"INFO": 1, "OPS": 2, "PARAMS": 3, "QUANT": 4, "DETECT": 5, "INT8_WEIGHTS": 6,
           "INT8_SCALES": 7, "FLOAT_WEIGHTS": 8, "NAMES": 9, "LAYER_OUTPUTS": 10}


def graph_ops(data):
    """The operation records of the export operation list."""
    max_id = 0
    for op in data["ops"]:
        for key in ("inp", "out"):
            v = op.get(key)
            if isinstance(v, list):
                max_id = max([max_id] + v)
            elif isinstance(v, int):
                max_id = max(max_id, v)
    next_id = max_id + 1
    ops = []

    def emit(kind, inputs, out, w=-1, b=-1, k=0, s=0, p=0, g=0, silu=False, index=0,
             w2=-1, b2=-1, w3=-1, b3=-1, heads=0, key_dim=0, head_dim=0, q=-1, q2=-1, q3=-1):
        ins = list(inputs) + [0] * (4 - len(inputs))
        ops.append((KINDS[kind], len(inputs), 1 if silu else 0, ins, out, w, b, k, s, p, g, index,
                    w2, b2, w3, b3, heads, key_dim, head_dim, q, q2, q3))

    for op in data["ops"]:  # the operation list written by the export (<name>_graph.json)
        kind = op["kind"]
        if kind == "conv":
            emit("Conv", [op["inp"]], op["out"], w=op["w"], b=op["b"], k=op["k"], s=op["s"],
                 p=op["p"], g=op["g"], silu=op["act"] == "silu", q=op.get("q", -1))
        elif kind == "add":
            emit("Add", op["inp"], op["out"])
        elif kind == "chunk":
            emit("Chunk", [op["inp"]], op["out"], index=op["index"])
        elif kind == "concat":
            ins = op["inp"]
            acc = ins[0]
            for j in range(1, len(ins)):
                target = op["out"] if j == len(ins) - 1 else next_id
                if target != op["out"]:
                    next_id += 1
                emit("Concat", [acc, ins[j]], target)
                acc = target
        elif kind == "maxpool":
            emit("MaxPool", [op["inp"]], op["out"], k=op["k"], s=op["s"], p=2 * op["p"])
        elif kind == "upsample2":
            emit("Upsample", [op["inp"]], op["out"])
        elif kind == "attention":
            emit("Attention", [op["inp"]], op["out"], w=op["qkv_w"], b=op["qkv_b"], w2=op["pe_w"],
                 b2=op["pe_b"], w3=op["proj_w"], b3=op["proj_b"], heads=op["heads"],
                 key_dim=op["key_dim"], head_dim=op["head_dim"], q2=op.get("q2", -1), q3=op.get("q3", -1))
        else:
            raise ValueError(kind)
    return ops, next_id + 1  # the tensor count


def encode_op(rec):
    kind, n_in, silu, ins, out, w, b, k, s, p, g, index, w2, b2, w3, b3, heads, key_dim, head_dim, q, q2, q3 = rec
    return struct.pack("<BBBB22i", kind, n_in, silu, 0, *ins, out, w, b, k, s, p, g, index, w2, b2, w3, b3,
                       heads, key_dim, head_dim, q, q2, q3) + struct.pack("<I", 0)


def section(name, payload):
    return SECTION[name], payload


def build(family, folder, per_tensor):
    base = family + "n"
    data = json.load(open(os.path.join(folder, base + "_graph.json")))
    ops, tensor_count = graph_ops(data)
    quant = data["quant_per_tensor"] if per_tensor else data["quant"]
    if per_tensor and not quant:
        raise SystemExit("yolo_make_tym: the export has no per-tensor scales (export with a photo)")
    det = data["detect"]
    names = data.get("names")
    if not names:
        raise SystemExit("yolo_make_tym: %s_graph.json has no class names (export again)" % base)
    if len(names) != det["classes"]:
        raise SystemExit("yolo_make_tym: %s needs %d class names, found %d" % (base, det["classes"], len(names)))
    name_bytes = base.encode()

    info = struct.pack("<B", len(name_bytes)) + name_bytes + struct.pack(
        "<6I", INPUT_SIZE, tensor_count, data["input_id"], len(ops), len(data["params"]), len(quant))
    ops_bytes = b"".join(encode_op(r) for r in ops)
    params = b""
    for p in data["params"]:
        dims = p["shape"] + [0] * (4 - len(p["shape"]))
        params += struct.pack("<iI4i", p["offset"], len(p["shape"]), *dims)
    quant_bytes = b"".join(struct.pack("<2i2f", e["w_offset"], e["s_offset"], e["in_scale"], e["out_scale"])
                           for e in quant)
    levels = len(det["strides"])
    detect = struct.pack("<3I", levels, det["classes"], det["reg_max"])
    for (box, cls), stride in zip(det["heads"], det["strides"]):
        detect += struct.pack("<3i", box, cls, stride)
    int8 = open(os.path.join(folder, base + "_int8_weights.bin"), "rb").read()
    scales = open(os.path.join(folder, base + "_int8_scales.bin"), "rb").read()
    floats = open(os.path.join(folder, base + "_graph_weights.bin"), "rb").read()
    names_bytes = struct.pack("<I", len(names))
    for n in names:
        b = n.encode()[:255]
        names_bytes += struct.pack("<B", len(b)) + b
    outputs = sorted((int(k), v) for k, v in data["outputs"].items())
    outputs_bytes = struct.pack("<I", len(outputs)) + b"".join(struct.pack("<2i", l, t) for l, t in outputs)

    sections = [("INFO", info), ("OPS", ops_bytes), ("PARAMS", params), ("QUANT", quant_bytes),
                ("DETECT", detect), ("INT8_WEIGHTS", int8), ("INT8_SCALES", scales),
                ("FLOAT_WEIGHTS", floats), ("NAMES", names_bytes), ("LAYER_OUTPUTS", outputs_bytes)]
    flags = 2 if per_tensor else 0
    flags |= 1  # HAS_FLOAT
    return pack_file(sections, flags), len(ops), tensor_count


def pack_file(sections, flags):
    header_size = 20 + 12 * len(sections)
    table = b""
    body = b""
    position = header_size
    for name, payload in sections:
        position = (position + 15) // 16 * 16
        body += b"\0" * (position - header_size - len(body))
        table += struct.pack("<3I", SECTION[name], position, len(payload))
        body += payload
        position += len(payload)
    head = b"TYM1" + struct.pack("<BBBBIII", 1, 0, 1, 0, header_size, flags, len(sections))
    file = head + table + body
    return file + struct.pack("<I", zlib.crc32(file) & 0xFFFFFFFF)


def main(argv):
    if len(argv) < 4 or argv[1] not in ("yolo11", "yolo26"):
        print(__doc__)
        return 2
    per_tensor = "--per-tensor" in argv[4:]
    blob, op_count, tensors = build(argv[1], argv[2], per_tensor)
    with open(argv[3], "wb") as f:
        f.write(blob)
    print("yolo_make_tym: %s: %d bytes (%s scales, %d ops, %d tensors)" % (
        argv[3], len(blob), "per-tensor" if per_tensor else "shared", op_count, tensors))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
