# TinyYolo

[![Arduino Library](https://img.shields.io/badge/Arduino-Library-blue.svg)](https://www.arduino.cc/reference/en/libraries/)
[![CMake](https://img.shields.io/badge/CMake-Supported-blue.svg)](CMakeLists.txt)
[![Header Only](https://img.shields.io/badge/Header-Only-blue.svg)](src/TinyYolo.h)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Introduction

TinyYolo runs YOLO object detection models on microcontroller boards. It is a header-only
C++ library: you include one file, and it works with Arduino `Print` and `Stream` objects.
It supports two kinds of model:

- **Darknet models** (YOLOv3-tiny, YOLOv4-tiny, YOLOv3 and YOLOv4 section sets) in the standard
  `.cfg`, `.names` and `.weights` format, stored in flash or on an SD card.
- **Ultralytics YOLO11 and YOLO26 (nano)** models, run as an anchor-free graph of primitive
  operations. Python tools export the checkpoint to int8 weights, a flash blob and a C++ operation
  table. See [Models](docs/Models.md#yolo11-ultralytics).

Weights can be stored as int8, which saves memory. The portable backend runs on every board, and
the ESP32 family has optimised backends:

- **ESP32-P4:** the int8 convolutions use Espressif's PIE assembly kernels (vendored, MIT). The
  YOLO11n graph takes about 1.5 s per 320x320 picture on this board.
- **ESP32-S3:** the int8 convolutions use ESP-NN (vendored in the library, no extra flags), and the linear 1x1 convolutions use our own exact kernel.
- **RP2040/RP2350 (Raspberry Pi Pico, RP2350B with PSRAM):** the int8 convolutions use CMSIS-NN (vendored, the default backend on these chips). The YOLO11n and YOLO26n graphs run from flash with the int8 weights in PSRAM.
- **Desktop:** the same graphs and operations run on the host PC (`desktop/`), with int8 (AVX2 on x86-64) and float32 paths.

Measured times and detections are in the [Performance](#performance) section, and the current state of
each board is in the [Status](#status) section. The classic ESP32
(4 MB flash) cannot hold the model blobs, so it is not supported for these models.

Object detection usually needs a Linux-class computer. TinyYolo targets the boards you
already use for sensors and cameras, so a detector can run next to the data it looks at.
TinyYolo fits well with two companion libraries:

- [TinyCamera](https://github.com/pschatzmann/TinyCamera) captures the frames that TinyYolo analyses.
- [TinyGPU](https://github.com/pschatzmann/TinyGPU) draws the results: it scales the frame to the
  display, outlines each detection and pushes the image to the LCD.

## Documentation

Detailed documentation is in the [docs](docs/) folder:

- [API](docs/API.md): `Detector`, images, logger, network descriptions and weight sources
- [Models](docs/Models.md): using an existing Darknet model from flash or SD, supported sections, memory
- [Training](docs/Training.md): training your own model with Darknet and making it fit the board
- [Quantization](docs/Quantization.md): int8 weights, memory savings and accuracy
- [Memory](docs/Memory.md): flash, PSRAM and internal RAM requirements of the YOLO models
- [Design](docs/Design.md): architecture, error handling and platform abstraction
- [Desktop](docs/Desktop.md): desktop build, the host tools (`desktop/`), the common viewer (`tinyyolo`: any model on the webcam or pictures), automated tests and sanitizers

## Status

Current state, as measured in this repository:

- **Working on every board:** the YOLO11n and YOLO26n int8 graphs find the same object (class 17,
  the horse) on the sample photo, in all four targets: the ESP32-P4, the ESP32-S3, the RP2350B and the
  desktop. Frame times are in the Performance table.
- **Accuracy:** the int8 scores on the photo are 0.818 to 0.905 against the reference (0.905 with the
  portable kernels). The SiLU activations are applied after the rounding, so the ESP-NN and CMSIS-NN
  paths differ from the reference by one step on up to about 4 % of the values.
- **Not supported on these boards:** YOLOv4-tiny (its float Darknet weights do not fit in the
  S3's or the RP2350B's flash or PSRAM at 320x320), and the classic ESP32 (4 MB flash).
- **Not measured yet:** the float32 graph on the ESP32-P4 (the 78.7 s in the table is from an earlier
  build), and the float32 graph on the RP2350B. The RP2350B has no example sketch in the repository yet;
  its test sketches are outside it.
- **Not committed:** the RP2350B support and the desktop `--size` option are in the working tree.

## Performance

Total time per picture on the ESP32-P4 (Waveshare ESP32-P4-WIFI6), the ESP32-S3, the RP2350B and the desktop (the host PC, `desktop/yolo/`, mean of three runs), with every model at the same
network input of 320x320. The photo (375x250) is letterboxed to 320x320. Each row is the mean of two
or three frames; the frame times varied by less than 3 %. The graph rows include the head.

| Model | Path | ESP32-P4 time | ESP32-S3 time | RP2350B time | Desktop time | Detections on the photo (class 17 = horse) |
|---|---|---:|---:|---:|---:|---|
| YOLO11n | graph, int8 (P4 assembly kernels; ESP-NN and exact 1x1 on the S3; CMSIS-NN on the RP2350B) | 1.5 s (head 7 ms) | 6.6 s (head 16 ms) | 18.9 s | 1.0 s | P4: score 0.818; S3: score 0.852; RP2350B: score 0.852; desktop: score 0.905 |
| YOLO11n | graph, float32 | 78.7 s (head 7 ms) | 63 s (head 16 ms) | not run | 2.2 s | score 0.833 |
| YOLO26n | graph, int8 (P4 assembly kernels; ESP-NN and exact 1x1 on the S3; CMSIS-NN on the RP2350B) | 1.9 s (head 6 ms) | 7.1 s (head 15 ms) | 15.8 s | 1.0 s | P4: score 0.818; S3: score 0.881; RP2350B: score 0.818; desktop: score 0.881 |
| YOLO26n | graph, float32 | not run | not run | not run | 2.0 s | score 0.891 |
| YOLOv4-tiny | Darknet model, int8 (weights only) | 13.8 s (head included) | does not fit (8 MB PSRAM) | does not fit (16 MB flash) | 10.2 s | P4: score 0.989; desktop: score 0.981 |
| YOLOv4-tiny | Darknet model, int8 with int8 activations (`--calibrate`) | not run | not run | not run | 3.3 s | desktop: score 0.985 |
| YOLOv4-tiny | Darknet model, float32 | not run | not run | not run | 7.5 s | score 0.979 |

The int8 convolutions on the S3 use the ESP-NN kernels, vendored in the library
(ESP-NN v1.5.0 as headers, so no extra flags are needed). ESP-NN rounds the requantization once, as the
reference does. Its activations are applied after the rounding, so the SiLU layers differ from the
reference by at most one step, on up to about 4 % of the values, and the linear 1x1 convolutions (the detection heads and the
neck projections) run on our own exact kernel, which decides the final scores. The YOLO11n S3 graph
finds the horse (class 17) at score 0.852, against 0.881 for the reference, in 6.6 s. The S3 numbers
are at 320x320 and come from three frames.

The RP2350B (SpotPear RP2350B-MINI-A, 8 MB PSRAM on GPIO 19, 16 MB flash) runs at 300 MHz with
`-Ofast` and PSRAM at 133 MHz. Its int8 convolutions use CMSIS-NN, which is the default for the RP2040
and RP2350. The weights of YOLO11n and YOLO26n are read from flash, and their int8 copies are in PSRAM.
Each RP2350B row is the mean of three frames.

The desktop tools run on the host PC with `DotProductBackend`. On x86-64 the int8 convolutions use AVX2 with exact int32 sums (the same result as the portable kernel). The Darknet models use weights-only int8 by default; `--calibrate` also quantizes the activations, which is the faster path. The desktop times were measured on a machine with other load (load average 1 to 5), so they vary by up to about 10 %. The desktop float32 and int8 scores are the reference values.

The P4 times use the P4 assembly kernels for all convolutions and depthwise convolutions. YOLOv4-tiny uses its Darknet model at 320x320 (the 192 config with the input size changed).

Espressif publishes faster INT8 YOLO11n timings for the ESP32-P4. We have not reproduced them: the
table above is measured on our board only.

## Installation in Arduino

You can download the library as ZIP and use *Sketch → Include Library → Add .ZIP Library...*.
Or you can clone the project into your Arduino libraries folder, e.g.

```
cd ~/Documents/Arduino/libraries
git clone <repository-url> TinyYolo
```

I recommend git, because you can update to the latest version with `git pull` in the project folder.

Examples are in the `examples/` folder:

- `CameraDetect`: live capture with TinyCamera and a Darknet model embedded in the sketch.
- `CameraStream`: streams camera frames to a desktop viewer.
- `DrawDetections`: boxes drawn on an LCD with TinyGPU.
- `Yolov4CameraStream`: YOLOv4-tiny (Darknet) on a camera board, streamed to the viewer.
- `Yolo11Graph`, `Yolo26Graph`: YOLO11n and YOLO26n on the ESP32-P4 with a sample photo.
- `Yolo11CameraStream`, `Yolo26CameraStream`: the same models on the P4 camera, streamed to the viewer.
