# Models

Open-source models, ready for TinyYolo. Each folder holds the model's files and a README. The
trained weights are too large for the repository: fetch them with the script of the model's
version (`tools/<version>_fetch_models.sh`).

| Folder | Model | Classes | Input | Path | Fetch |
|---|---|---|---|---|---|
| `yolov4-tiny/` | YOLOv4-tiny, trained on COCO | 80 | 416x416 (examples: 192x192) | Darknet (`.cfg`, `.weights`) | `tools/yolov4_fetch_models.sh` |
| `yolo11n/` | YOLO11n (Ultralytics), trained on COCO | 80 | 640x640 | graph (table and blob, exported) | `tools/yolo11/yolo11_fetch_models.sh` |
| `yolo26n/` | YOLO26n (Ultralytics), trained on COCO | 80 | 640x640 | graph (table and blob, exported) | `tools/yolo26/yolo26_fetch_models.sh` |

Use a YOLOv4-tiny model with the tinyyolo tool:

```bash
tools/yolov4_fetch_models.sh
cmake -S . -B build && cmake --build build --target tinyyolo
build/desktop/tinyyolo --model models/yolov4-tiny/yolov4-tiny --image photo.ppm --no-window
```

Licences: the Darknet code and the YOLOv4 model's licence text are in `yolov4-tiny/`
(`LICENSE-darknet`). The YOLO11 weights are AGPL-3.0 (see `yolo11n/README.md`). The COCO dataset,
which the classes come from, is CC BY 4.0.
