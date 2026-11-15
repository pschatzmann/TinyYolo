# Examples

The sketches in `examples/`. Each folder has a README with the steps: the model files, the build and the
upload. The desktop tool that shows the camera streams is `tinyyolo --serial` (see [Desktop](Desktop.md)).

- [`CameraDetect`](../examples/CameraDetect/): live capture with TinyCamera and a Darknet model embedded in the sketch.
- [`CameraStream`](../examples/CameraStream/): YOLO11n on the ESP32-S3-EYE camera, streamed to the desktop tool.
- [`DrawDetections`](../examples/DrawDetections/): boxes drawn on an LCD with TinyGPU.
- [`Yolov4CameraStream`](../examples/Yolov4CameraStream/): YOLOv4-tiny (Darknet) on a camera board, streamed to the desktop tool.
- [`Yolo11Graph`](../examples/Yolo11Graph/), [`Yolo26Graph`](../examples/Yolo26Graph/): YOLO11n and YOLO26n on the ESP32-P4 with a sample photo.
- [`Yolo11CameraStream`](../examples/Yolo11CameraStream/), [`Yolo26CameraStream`](../examples/Yolo26CameraStream/): the same models on the P4 camera, streamed to the desktop tool.
