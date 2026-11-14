# YOLO26n

Ultralytics YOLO26n (nano), trained on COCO (80 classes), input 640x640. It runs on the TinyYolo
graph path (`tinyyolo::GraphYolo11`, which decodes any anchor-free graph model), not from a Darknet
file. Compared with YOLO11n it has no DFL: the box head predicts four distances directly
(`reg_max` 1), the SPPF has a residual connection, and one C3k2 block of the neck holds a PSA
attention block.

| File | Content | How it is made |
|---|---|---|
| `yolo26n.pt` | the Ultralytics checkpoint (5.5 MB) | downloaded by `tools/yolo26/yolo26_fetch_models.sh` (checksum checked) |
| `yolo26n_graph.inc` | the operation table: `kOps`, `kParams`, `kQuant`, the detect heads, `kModel` | exported from `yolo26n.pt` |
| `export/yolo26n_board.bin` | the board blob: float parameters, int8 weights, weight scales (12067648 bytes) | exported; flashed to the `model` partition |
| `yolo26n.names` | the 80 class names | from the checkpoint |

Get the weights and the export (needs Python with the `ultralytics` package):

```sh
PYTHON=python3 tools/yolo26/yolo26_fetch_models.sh
```

The int8 activation scales are calibrated on one image: by default the test photo of the examples.
Set `CALIBRATION_IMAGE` to another image (PPM, PNG or JPEG) for other scenes.

The examples use the table and the blob:

- `examples/Yolo26Graph`: the sample photo on the ESP32-P4 (flash the blob first).
- `examples/Yolo26CameraStream`: the P4 camera, streamed to the desktop viewer.
- `yolo26_detect` and `yolo26_check` (built by CMake from `desktop/yolo/`): the same model on the desktop.

Licence: the Ultralytics code and weights are AGPL-3.0 (https://github.com/ultralytics/ultralytics).
The COCO dataset, which the classes come from, is CC BY 4.0.
