#!/usr/bin/env python3
"""Converts an Ultralytics YOLO checkpoint (.pt) for TinyYolo: YOLO11 and YOLO26 (the nano models
and the other scales with the same blocks).

usage: yolo_export.py yolo11n.pt out_dir [photo.jpg]

Writes into out_dir:
  <name>_graph_weights.bin   float32 parameters, in the order the operations use them
  <name>_graph.json          the operation list (for reference and for yolo_gen_graph.py)
  <name>_board.bin           the three weight files concatenated: the blob for the P4 examples
  <name>.inc                 the C++ table (namespace yolo11 or yolo26), for the desktop tools
  input_640.bin, ref_layerNN_0.bin, ref_layer23_0.bin
                             the letterboxed photo and the PyTorch outputs, to check the C++ graph
Needs the ultralytics package (and torch). Batch-norm is folded into the convolutions.
"""
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.modules.block import Bottleneck, C3k, PSABlock
from ultralytics.nn.modules.conv import Conv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yolo_gen_graph  # noqa: E402


class Graph:
    """The network as primitive operations, with the parameters they use."""

    def __init__(self):
        self.ops = []
        self.params = []
        self.n = 0

    def new(self):
        self.n += 1
        return self.n - 1

    def param(self, t):
        self.params.append(t.detach().clone().float())
        return len(self.params) - 1

    def conv(self, mod, x):
        c = mod.conv
        out = self.new()
        act = "silu" if isinstance(mod.act, nn.SiLU) else "linear"
        self.ops.append(dict(kind="conv", inp=x, out=out, w=self.param(c.weight), b=self.param(c.bias),
                             k=c.kernel_size[0], s=c.stride[0], p=c.padding[0], g=c.groups, act=act))
        return out

    def conv_plain(self, mod, x):
        out = self.new()
        self.ops.append(dict(kind="conv", inp=x, out=out, w=self.param(mod.weight), b=self.param(mod.bias),
                             k=mod.kernel_size[0], s=mod.stride[0], p=mod.padding[0], g=mod.groups,
                             act="linear"))
        return out

    def binop(self, kind, a, b):
        out = self.new()
        self.ops.append(dict(kind=kind, inp=[a, b], out=out))
        return out

    def chunk(self, x, index):
        out = self.new()
        self.ops.append(dict(kind="chunk", inp=x, out=out, index=index))
        return out

    def concat(self, xs):
        out = self.new()
        self.ops.append(dict(kind="concat", inp=list(xs), out=out))
        return out

    def maxpool(self, x, k, s, p):
        out = self.new()
        self.ops.append(dict(kind="maxpool", inp=x, out=out, k=k, s=s, p=p))
        return out

    def upsample(self, x):
        out = self.new()
        self.ops.append(dict(kind="upsample2", inp=x, out=out))
        return out

    def attention(self, mod, x):
        out = self.new()
        self.ops.append(dict(kind="attention", inp=x, out=out,
                             qkv_w=self.param(mod.qkv.conv.weight), qkv_b=self.param(mod.qkv.conv.bias),
                             pe_w=self.param(mod.pe.conv.weight), pe_b=self.param(mod.pe.conv.bias),
                             proj_w=self.param(mod.proj.conv.weight), proj_b=self.param(mod.proj.conv.bias),
                             heads=mod.num_heads, key_dim=mod.key_dim, head_dim=mod.head_dim))
        return out

    def psa_block(self, mod, x):
        a = self.attention(mod.attn, x)
        x = self.binop("add", x, a) if mod.add else a
        f = self.conv(mod.ffn[1], self.conv(mod.ffn[0], x))
        return self.binop("add", x, f) if mod.add else f

    def bottleneck(self, mod, x):
        y = self.conv(mod.cv2, self.conv(mod.cv1, x))
        return self.binop("add", x, y) if mod.add else y


def c3k(g, mod, x):
    a = g.conv(mod.cv1, x)
    b = g.conv(mod.cv2, x)
    for blk in mod.m:
        a = g.bottleneck(blk, a)
    return g.conv(mod.cv3, g.concat([a, b]))


def detect_branch(g, layer, lvl, src):
    """Raw box and class outputs of one scale (before the DFL decode)."""
    box = g.conv(layer.cv2[lvl][0], src)
    box = g.conv(layer.cv2[lvl][1], box)
    box = g.conv_plain(layer.cv2[lvl][2], box)
    cls = src
    for part in layer.cv3[lvl][:2]:
        cls = g.conv(part[0], cls)
        cls = g.conv(part[1], cls)
    cls = g.conv_plain(layer.cv3[lvl][2], cls)
    return box, cls


def build(net):
    """The flattened graph of a fused Ultralytics model, and the tensor id of each layer."""
    g = Graph()
    x = g.new()  # network input
    out_of = {}
    cur = x
    for i, layer in enumerate(net.model):
        name = type(layer).__name__
        if name == "Conv":
            cur = g.conv(layer, cur)
        elif name == "C3k2":
            y0 = g.conv(layer.cv1, cur)
            ys = [g.chunk(y0, 0), g.chunk(y0, 1)]
            for blk in layer.m:
                if isinstance(blk, Bottleneck):
                    ys.append(g.bottleneck(blk, ys[-1]))
                elif isinstance(blk, C3k):
                    ys.append(c3k(g, blk, ys[-1]))
                elif isinstance(blk, nn.Sequential) and all(isinstance(b, (Bottleneck, PSABlock)) for b in blk):
                    y = ys[-1]
                    for part in blk:
                        y = g.bottleneck(part, y) if isinstance(part, Bottleneck) else g.psa_block(part, y)
                    ys.append(y)
                else:
                    raise ValueError(f"unsupported block {type(blk).__name__}")
            cur = g.conv(layer.cv2, g.concat(ys))
        elif name == "SPPF":
            x_in = cur
            y0 = g.conv(layer.cv1, cur)
            ys = [y0]
            for _ in range(getattr(layer, "n", 3)):
                ys.append(g.maxpool(ys[-1], 5, 1, 2))
            cur = g.conv(layer.cv2, g.concat(ys))
            if getattr(layer, "add", False):
                cur = g.binop("add", x_in, cur)
        elif name == "C2PSA":
            y0 = g.conv(layer.cv1, cur)
            a, b = g.chunk(y0, 0), g.chunk(y0, 1)
            for blk in layer.m:
                b = g.psa_block(blk, b)
            cur = g.conv(layer.cv2, g.concat([a, b]))
        elif name == "Upsample":
            cur = g.upsample(cur)
        elif name == "Concat":
            cur = g.concat([cur if j == -1 else out_of[j] for j in layer.f])
        elif name == "Detect":
            cur = [detect_branch(g, layer, lvl, out_of[j]) for lvl, j in enumerate(layer.f)]
        else:
            raise ValueError(f"unsupported layer {name}")
        out_of[i] = cur
    return g, out_of, x


def attention_parts(x, g, op):
    """Float attention, as the C++ Graph computes it (see Graph.h). Returns the qkv convolution
    output and the attention output before the projection."""
    B, C, H, W = x.shape
    N = H * W
    qkv_full = F.conv2d(x, g.params[op["qkv_w"]], g.params[op["qkv_b"]])
    heads, kd, hd = op["heads"], op["key_dim"], op["head_dim"]
    qkv = qkv_full.reshape(B, heads, kd * 2 + hd, N)
    q, k, v = qkv.split([kd, kd, hd], dim=2)
    attn = ((q * kd ** -0.5).transpose(-2, -1) @ k).softmax(dim=-1)
    y = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W)
    y = y + F.conv2d(v.reshape(B, C, H, W), g.params[op["pe_w"]], g.params[op["pe_b"]], padding=1, groups=C)
    return qkv_full, y


def attention_op(x, g, op):
    """Float attention, as the C++ Graph computes it (see Graph.h)."""
    _, y = attention_parts(x, g, op)
    return F.conv2d(y, g.params[op["proj_w"]], g.params[op["proj_b"]])


def run_ops(g, x):
    """Runs the operations in float (torch). Returns every tensor by id."""
    t = {0: x}
    for op in g.ops:
        k = op["kind"]
        if k == "conv":
            y = F.conv2d(t[op["inp"]], g.params[op["w"]], g.params[op["b"]], stride=op["s"],
                         padding=op["p"], groups=op["g"])
            t[op["out"]] = F.silu(y) if op["act"] == "silu" else y
        elif k == "add":
            t[op["out"]] = t[op["inp"][0]] + t[op["inp"][1]]
        elif k == "chunk":
            a = t[op["inp"]]
            c = a.shape[1] // 2
            t[op["out"]] = a[:, :c] if op["index"] == 0 else a[:, c:]
        elif k == "concat":
            t[op["out"]] = torch.cat([t[i] for i in op["inp"]], dim=1)
        elif k == "maxpool":
            t[op["out"]] = F.max_pool2d(t[op["inp"]], op["k"], op["s"], op["p"])
        elif k == "upsample2":
            t[op["out"]] = F.interpolate(t[op["inp"]], scale_factor=2, mode="nearest")
        elif k == "attention":
            t[op["out"]] = attention_op(t[op["inp"]], g, op)
        else:
            raise ValueError(k)
    return t


def letterbox_tensor(photo_path, size=640):
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    return torch.from_numpy(np.asarray(canvas).astype("float32") / 255.0).permute(2, 0, 1)[None]


def pow2_scale(x):
    """The smallest power of two that is at least x (so that x / scale does not clip)."""
    return float(2.0 ** math.ceil(math.log2(x)))


def quantize(g, photo_path, shared=True):
    """Calibrates the activations on the photo and quantizes the standard convolutions to int8.

    Each convolution (groups == 1) gets an input and an output scale and per-output-channel weight
    scales. All scales are powers of two (rounded up from max / 127), so the int8 kernels can
    requantize with an integer shift. Its weights are written in OHWI order, as the int8 kernels read them.
    Returns (int8 weights, float scales, per-operation quantization entries)."""
    with torch.no_grad():
        t = run_ops(g, letterbox_tensor(photo_path))
    maxabs = {i: float(v.abs().max()) for i, v in t.items()}
    # The inputs and the output of a concatenation share one scale (the largest of them), so that the
    # int8 concatenation is a copy of the bytes. Union-find over the tensors of each concatenation.
    parent = {}

    def find(a):
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    for op in g.ops:
        if op["kind"] in ("chunk", "maxpool", "upsample2"):
            # A channel slice, a max pooling and a nearest upsampling keep the scale of their input.
            parent[find(op["out"])] = find(op["inp"])
        if op["kind"] == "add":
            # A sum of two int8 tensors needs both inputs and the output at one scale.
            for other in op["inp"][1:] + [op["out"]]:
                ra, rb = find(op["inp"][0]), find(other)
                if ra != rb:
                    parent[rb] = ra
        if op["kind"] == "concat":
            ids = list(op["inp"]) + [op["out"]]
            for other in ids[1:]:
                ra, rb = find(ids[0]), find(other)
                if ra != rb:
                    parent[rb] = ra
    group_max = {}
    for tid, m in maxabs.items():
        r = find(tid)
        group_max[r] = max(group_max.get(r, 0.0), m)

    def tscale(tid):
        """Power-of-two scale of tensor tid. With shared=True the tensors of a concatenation, an add,
        a chunk, a max pooling or an upsampling share one scale (see above); with shared=False each
        tensor has its own."""
        if not shared:
            return pow2_scale(max(maxabs.get(tid, 1e-6), 1e-6) / 127.0)
        return pow2_scale(max(group_max.get(find(tid), maxabs.get(tid, 1e-6)), 1e-6) / 127.0)
    q_w, q_s, entries = [], [], []
    w_off, s_off = 0, 0
    for op in g.ops:
        if op["kind"] == "attention":
            # The two 1x1 convolutions of the attention (qkv and projection), int8 per output channel.
            x_t = t[op["inp"]]
            qkv_full, y = attention_parts(x_t, g, op)
            for key, w_key, in_val, out_val in (
                ("q2", "qkv_w", tscale(op["inp"]), pow2_scale(float(qkv_full.abs().max()) / 127.0)),
                ("q3", "proj_w", pow2_scale(float(y.abs().max()) / 127.0), tscale(op["out"])),
            ):
                rows = g.params[op[w_key]].reshape(g.params[op[w_key]].shape[0], -1)
                ws = torch.tensor([pow2_scale(float(m) / 127.0) for m in rows.abs().max(dim=1).values])
                ws = torch.where(ws > 0, ws, torch.ones_like(ws))
                qw = torch.clamp(torch.round(rows / ws[:, None]), -127, 127).to(torch.int8)
                q_w.append(qw.numpy().reshape(-1))
                q_s.append(ws.numpy().astype("<f4"))
                op[key] = len(entries)
                entries.append({"w_offset": w_off, "s_offset": s_off,
                                "in_scale": in_val, "out_scale": out_val})
                w_off += qw.numel()
                s_off += ws.numel()
            continue
        if op["kind"] != "conv":
            continue
        depthwise = op["g"] > 1 and op["g"] == g.params[op["w"]].shape[0]
        if op["g"] != 1 and not depthwise:
            continue
        in_scale = tscale(op["inp"])
        out_scale = tscale(op["out"])
        if depthwise:
            # One filter per channel ([C, 1, K, K]). The P4 depthwise kernel takes one shift for the
            # whole layer, so the layer has one power-of-two weight scale; the rows are [C][K*K].
            rows = g.params[op["w"]].reshape(g.params[op["w"]].shape[0], -1)
            s_layer = pow2_scale(max(float(rows.abs().max()), 1e-12) / 127.0)
            ws = torch.full((rows.shape[0],), s_layer)
        else:
            W = g.params[op["w"]].permute(0, 2, 3, 1).contiguous()  # OIHW -> OHWI
            rows = W.reshape(W.shape[0], -1)
            ws = torch.tensor([pow2_scale(float(m) / 127.0) for m in rows.abs().max(dim=1).values])
            ws = torch.where(ws > 0, ws, torch.ones_like(ws))
        qw = torch.clamp(torch.round(rows / ws[:, None]), -127, 127).to(torch.int8)
        q_w.append(qw.numpy().reshape(-1))
        q_s.append(ws.numpy().astype("<f4"))
        op["q"] = len(entries)
        entries.append({"w_offset": w_off, "s_offset": s_off, "in_scale": in_scale, "out_scale": out_scale})
        w_off += qw.numel()
        s_off += ws.numel()
    return np.concatenate(q_w).astype(np.int8), np.concatenate(q_s).astype("<f4"), entries


def export(net, out_dir, base, photo_path=None):
    g, out_of, _ = build(net)
    quant_entries = []
    quant_per_tensor = []
    if photo_path is not None:
        q_w, q_s, quant_entries = quantize(g, photo_path, shared=True)
        # The weights do not depend on the activation scales: only the table of entries differs. The
        # second pass gives the per-tensor table (the same operation indices).
        _, _, quant_per_tensor = quantize(g, photo_path, shared=False)
        q_w.tofile(os.path.join(out_dir, f"{base}_int8_weights.bin"))
        q_s.tofile(os.path.join(out_dir, f"{base}_int8_scales.bin"))
    offsets, off = [], 0
    with open(os.path.join(out_dir, f"{base}_graph_weights.bin"), "wb") as f:
        for p in g.params:
            arr = p.numpy().astype("<f4")
            f.write(arr.tobytes())
            offsets.append({"offset": off, "shape": list(arr.shape)})
            off += arr.size
    detect = [v for v in out_of.values() if isinstance(v, list)][-1]
    head = net.model[-1]
    doc = {"params": offsets, "ops": g.ops, "input_id": 0,
           "outputs": {str(k): v for k, v in out_of.items() if isinstance(v, int)},
           "quant": quant_entries,
           "quant_per_tensor": quant_per_tensor if photo_path is not None else [],
           "detect": {"heads": [[int(b), int(c)] for b, c in detect],
                      "strides": [int(s) for s in head.stride.tolist()],
                      "classes": int(head.nc), "reg_max": int(head.reg_max)}}
    json_path = os.path.join(out_dir, f"{base}_graph.json")
    with open(json_path, "w") as f:
        json.dump(doc, f, indent=1)
    family = re.match(r"yolo\d+", base)
    namespace = family.group(0) if family else "yolo11"
    yolo_gen_graph.main(json_path, os.path.join(out_dir, f"{base}.inc"), namespace)
    print(f"weights: {off} floats in {len(g.params)} tensors; {len(g.ops)} operations")


def reference(net, out_dir, photo_path, size=640):
    """Letterboxes the photo (pad 114) and saves the input and the outputs of every layer."""
    img = Image.open(photo_path).convert("RGB")
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (left, top))
    x = torch.from_numpy(np.asarray(canvas).astype("float32") / 255.0).permute(2, 0, 1)[None]
    x.numpy().astype("<f4").tofile(os.path.join(out_dir, "input_640.bin"))

    outputs = {}
    hooks = []
    for i, layer in enumerate(net.model):
        def hook(mod, inp, out, i=i):
            found = []

            def collect(o):
                if torch.is_tensor(o):
                    found.append(o.detach().cpu().numpy())
                elif isinstance(o, (list, tuple)):
                    for v in o:
                        collect(v)
                elif isinstance(o, dict):
                    for k in sorted(o):
                        collect(o[k])
            collect(out)
            outputs[i] = found
        hooks.append(layer.register_forward_hook(hook))
    with torch.no_grad():
        net(x)
    for hk in hooks:
        hk.remove()
    for i, arrs in outputs.items():
        for k, arr in enumerate(arrs):
            arr.astype("<f4").tofile(os.path.join(out_dir, f"ref_layer{i:02d}_{k}.bin"))
    print(f"reference: input and {len(outputs)} layers; letterbox scale {scale:.4f}, pad {left} {top}")


def write_board_blob(out_dir, base):
    """Writes <base>_board.bin: the float parameters, the int8 weights and the weight scales, in
    this order. The ESP32-P4 example (examples/Yolo11Graph) copies it to the "model" partition."""
    names = [f"{base}_graph_weights.bin", f"{base}_int8_weights.bin", f"{base}_int8_scales.bin"]
    path = os.path.join(out_dir, f"{base}_board.bin")
    with open(path, "wb") as dst:
        for name in names:
            with open(os.path.join(out_dir, name), "rb") as src:
                dst.write(src.read())
    print(f"board blob: {path} ({os.path.getsize(path)} bytes)")
    return path


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    weights_pt, out_dir = argv[1], argv[2]
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(weights_pt))[0]
    net = YOLO(weights_pt).model.fuse().eval().float()
    photo = argv[3] if len(argv) > 3 else None
    export(net, out_dir, base, photo)
    write_board_blob(out_dir, base)
    if photo is not None:
        reference(net, out_dir, photo)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
