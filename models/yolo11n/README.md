# YOLO11n

Ultralytics YOLO11n (nano), trained on COCO (80 classes), input 640x640. It runs on the TinyYolo
graph path (`tinyyolo::GraphYolo11`), not from a Darknet file.

| File | Content | How it is made |
|---|---|---|
| `yolo11n.pt` | the Ultralytics checkpoint (5.6 MB) | downloaded by `tools/yolo11/yolo11_fetch_models.sh` (checksum checked) |
| `yolo11n_graph.inc` | the operation table: `kOps`, `kParams`, `kQuant`, the detect heads, `kModel` | exported from `yolo11n.pt` |
| `export/yolo11n_board.bin` | the board blob: float parameters, int8 weights, weight scales (13104288 bytes) | exported; flashed to the `model` partition |
| `export/` (other files) | the intermediate export (graph weights, operation list, table) | exported |

The int8 activation scales are calibrated on one image: by default the test photo of the examples
(`examples/Yolo11Graph/photo.rgb`). For other scenes, set `CALIBRATION_IMAGE` to a typical image
(PPM, PNG or JPEG). The export needs the calibration image.

Get the weights and the export (needs Python with the `ultralytics` package):

```sh
PYTHON=python3 tools/yolo11/yolo11_fetch_models.sh
```

The examples use the table and the blob:

- `examples/Yolo11Graph`: the sample photo on the ESP32-P4 (flash the blob first, see the example's README).
- `examples/Yolo11CameraStream`: the P4 camera, streamed to the desktop viewer.
- `yolo11_detect`: the same model on the desktop, `yolo11_detect models/yolo11n/export image.ppm` (built by CMake from `desktop/yolo/yolo_detect.cpp`).

Licence: the Ultralytics code and weights are AGPL-3.0 (https://github.com/ultralytics/ultralytics).
The COCO dataset, which the classes come from, is CC BY 4.0.
