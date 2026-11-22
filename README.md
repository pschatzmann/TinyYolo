# TinyYolo

[![Arduino Library](https://img.shields.io/badge/Arduino-Library-blue.svg)](https://www.arduino.cc/reference/en/libraries/)
[![CMake](https://img.shields.io/badge/CMake-Supported-blue.svg)](CMakeLists.txt)
[![Header Only](https://img.shields.io/badge/Header-Only-blue.svg)](src/TinyYolo.h)
[![License: Apache](https://img.shields.io/badge/License-Apache-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

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

- **ESP32-P4:** the int8 convolutions use Espressif's PIE assembly kernels (vendored, Apache-2.0). The
  YOLO11n graph takes about 1.5 s per 320x320 picture on this board.
- **ESP32-S3:** the int8 convolutions use ESP-NN (vendored in the library, no extra flags), and the linear 1x1 convolutions use our own exact kernel.
- **RP2040/RP2350 (Raspberry Pi Pico, RP2350B with PSRAM):** the int8 convolutions use CMSIS-NN (vendored, the default backend on these chips). The YOLO11n and YOLO26n graphs run from flash with the int8 weights in PSRAM.
- **Desktop:** the same graphs and operations run on the host PC (`desktop/`), with int8 (AVX2 on x86-64) and float32 paths.

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
- [Setup](docs/Setup.md): what to install for the desktop tools, the model export and the boards
- [Model format](docs/ModelFormat.md): the single-file `.tym` model
- [Examples](docs/Examples.md): the sketches in `examples/` and what each one runs
- [Status](docs/Status.md): the measured state of each board
- [Performance](docs/Performance.md): measured times and detections
- [Desktop](docs/Desktop.md): desktop build, the host tools (`desktop/`), the common viewer (`tinyyolo`: any model on the webcam or pictures), automated tests and sanitizers

## Installation in Arduino

You can download the library as ZIP and use *Sketch → Include Library → Add .ZIP Library...*.
Or you can clone the project into your Arduino libraries folder, e.g.

```
cd ~/Documents/Arduino/libraries
git clone <repository-url> TinyYolo
```

I recommend git, because you can update to the latest version with `git pull` in the project folder.

The sketches are in the `examples/` folder, described in [Examples](docs/Examples.md).
