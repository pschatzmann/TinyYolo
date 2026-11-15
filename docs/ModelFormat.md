# Single-file model format (proposal)

Status: **implemented** (format version 1.0). The name `.tym` is the file extension in use. The
current models use several files per model (see [Models](Models.md)); this format puts a whole model
in one file, so the board can load it from one flash partition and the desktop tool can load it from
one path.

Code: `src/TinyYolo/app/ModelFile.h` (reader), `src/TinyYolo/app/ModelFileWriter.h` (writer),
`src/TinyYolo/app/ModelFilePartition.h` (ESP32 partition loader), `tools/yolo/yolo_make_tym.py` (written by
the export, or from an export folder), `tinyyolo --tym <file>` (desktop), and `tests/host/test_model_file.cpp` (the tests).

## Goals

- One file per model: graph, quantization, int8 weights, class names, and optionally float weights.
- The board reads the file in place, from a flash partition, with no copy into RAM. Every section
  starts on a 16-byte boundary, so the int8 kernels can use the weights directly.
- The graph is data, not generated C code. A new model needs no recompile of the sketch.
- The desktop reads the same file and can use the float weights for the float32 path.

## Decisions (defaults for the open points)

| Question | Decision |
|---|---|
| Float weights in the same file? | Yes, and required for the YOLO models: `GraphYolo11` needs the float parameters for the layers that stay float (the detection head and the other layers without int8 scales). A file without `HAS_FLOAT` is valid, but the graph cannot run from it. Storing only the float parameters that are used would shrink the file; that is a later change. |
| Graph as data or compiled table? | Data. The ops are stored as fixed-size records that match `GraphOp` (`src/TinyYolo/graph/Graph.h`). |
| Version handling | `major.minor`. A reader rejects a different major version with a clear error. A reader accepts a higher minor version and skips the sections it does not know. |
| Name | `.tym`, until you choose another. |

## File layout

All integers and floats are little-endian. Offsets are from the start of the file.

```
+0x00  header            (magic, version, section table)
+....  section 0 ...     (each section 16-byte aligned)
...
end-4  CRC32            (over all bytes before it)
```

### Header

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `TYM1` |
| 4 | 2 | major version (1) |
| 6 | 2 | minor version (0) |
| 8 | 4 | header size in bytes, including the section table |
| 12 | 4 | flags: bit 0 `HAS_FLOAT`, bit 1 `PER_TENSOR_QUANT` |
| 16 | 4 | number of sections |
| 20 | 12 × n | section table: type (u32), offset (u32), size (u32) |

### Section types

| Type | Name | Content |
|---|---|---|
| 1 | `INFO` | model name (u8 length, then the bytes), input size (u32), tensor count (u32), input tensor id (u32), number of ops (u32), number of params (u32), number of quant entries (u32) |
| 2 | `OPS` | `op_count` records of 96 bytes (see below) |
| 3 | `PARAMS` | `param_count` records of 24 bytes: offset (i32, in floats), rank (u32), dims[4] (i32) |
| 4 | `QUANT` | `quant_count` records of 16 bytes: w_offset (i32, int8 values), s_offset (i32, floats), in_scale (f32), out_scale (f32) |
| 5 | `DETECT` | detect levels (u32, at most 3), number of classes (u32), box bins per side `reg_max` (u32: 16 for DFL, 1 for direct), then per level: box tensor id (i32), class tensor id (i32), stride (i32, network-input pixels per cell) |
| 6 | `INT8_WEIGHTS` | int8 convolution weights, OHWI, 16-byte aligned |
| 7 | `INT8_SCALES` | per-output-channel weight scales (f32) |
| 8 | `FLOAT_WEIGHTS` | float parameters (f32), only if `HAS_FLOAT` |
| 10 | `LAYER_OUTPUTS` (optional) | number of pairs (u32), then per pair: export layer (i32), tensor id (i32). Used by the desktop checks to compare each layer with its reference |
| 9 | `NAMES` | number of names (u32, must equal the number of classes), then each name as a length-prefixed string (u8 length, UTF-8, at most 255 bytes) |

`PER_TENSOR_QUANT` selects the per-tensor activation scales in `QUANT`. Both variants of the current
export (`kQuant` and `kQuantPerTensor`) fit in this one section type, since the file holds one of them.

### OPS record (96 bytes)

Matches `GraphOp` field by field. Unused fields are 0 or -1, as in the struct.

| Offset | Size | Field |
|---|---|---|
| 0 | 1 | kind (`GraphOpKind`: 0 Conv, 1 Add, 2 Chunk, 3 Concat, 4 MaxPool, 5 Upsample, 6 Attention) |
| 1 | 1 | n_in |
| 2 | 1 | flags: bit 0 SiLU |
| 3 | 1 | reserved (0) |
| 4 | 16 | in[4] (i32) |
| 20 | 4 | out |
| 24 | 4 | w |
| 28 | 4 | b |
| 32 | 4 | k |
| 36 | 4 | s |
| 40 | 4 | p |
| 44 | 4 | g |
| 48 | 4 | index |
| 52 | 4 | w2 |
| 56 | 4 | b2 |
| 60 | 4 | w3 |
| 64 | 4 | b3 |
| 68 | 4 | heads |
| 72 | 4 | key_dim |
| 76 | 4 | head_dim |
| 80 | 4 | q |
| 84 | 4 | q2 |
| 88 | 4 | q3 |
| 92 | 4 | reserved (0) |

## Reading the file

On the board, the file is in a flash partition. The reader maps it (as `ModelFilePartition` does) and checks, before using anything:

1. the magic and the major version,
2. that every section lies inside the file and starts on a 16-byte boundary,
3. the CRC32,
4. that each op refers to tensor ids below `tensor_count`, and each param and quant index is in range.

Any failure returns an error status and does not start the model. The desktop reads the same file
into memory and runs the same checks.

## Migration

- The current exports stay as they are until the exporter writes `.tym` files.
- `tools/yolo/yolo_make_tym.py` converts an export folder (the operation list, the weights and the names) into a
  `.tym` file. The export runs it, so the export writes the `.tym` next to the export folder. The Python exporter is unchanged.
- The board example `CameraStream` loads the `.tym` file from the `model` partition. The other examples
  still use the current blobs and tables until they are moved to `.tym`.

## Decisions on the open items

- **Detection heads:** YOLO11 and YOLO26 are anchor-free, so there are no anchors in the file. The
  `DETECT` section carries what `GraphModel` already has: levels, classes, `reg_max`, and per level
  the box and class tensor ids and the stride. A model with anchors would need a new section type,
  which a minor version can add.
- **Class names:** one name per class, each at most 255 bytes. The count must equal the number of
  classes, and the reader rejects a mismatch.
- **Calibration photo:** not in the file. The export checks use separate reference files, and the
  board never needs them. The model file stays the deployable part only.
