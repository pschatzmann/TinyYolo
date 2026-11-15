# YOLOv4-tiny

YOLOv4-tiny (Darknet), trained on COCO (80 classes). The network in `yolov4-tiny.cfg` takes 416x416 pictures;
the `Yolov4CameraStream` example uses a 192x192 version of the network, and the desktop tool can run other input
sizes with `--size` (see the example's README).
It runs on the TinyYolo Darknet path (`TinyYoloModel`), not as a single-file `.tym` model.

| File | Content | How it is made |
|---|---|---|
| `yolov4-tiny.cfg` | the Darknet network description (80 classes) | from the Darknet project (AlexeyAB/darknet) |
| `yolov4-tiny.names` | the 80 class names, one per line | from the COCO dataset |
| `yolov4-tiny.weights` | the trained Darknet weights (24251276 bytes) | downloaded by `tools/fetch_models.py yolov4-tiny` (checksum checked) |
| `LICENSE-darknet` | the Darknet licence | from the Darknet project |

Get the weights (if they are missing or do not match the checksum):

```sh
python tools/fetch_models.py yolov4-tiny
```

Use the model with the desktop tool (the `.cfg`, `.names` and `.weights` files are read by their base name):

```sh
cmake -S . -B build && cmake --build build --target tinyyolo
build/desktop/tinyyolo --model models/yolov4-tiny/yolov4-tiny --image photo.ppm --no-window
```

On the board, the `Yolov4CameraStream` example reads the same network (see its README for the flash steps).

Licences: the Darknet code is public domain (`LICENSE-darknet`). The COCO dataset, which the classes come
from, is CC BY 4.0 (https://cocodataset.org).
