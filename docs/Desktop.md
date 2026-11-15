# Desktop build, tools and tests

TinyYolo builds and runs on Linux, macOS and Windows with CMake and a C++11 compiler.

## Build and run the tests

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## What the tests cover

`tests/host/test_tinyyolo.cpp` (about 670 checks):

- logger levels, output format and the static instance
- tensors, activations (including Mish), convolution, pooling, upsampling, concatenation and
  the residual add
- image formats, letterboxing and box drawing
- the decoder (YOLOv5 and Darknet styles) and NMS
- model construction, weight loading, activation memory reuse and error paths
- per-layer and int8 precision: quantizer, int8 convolution and whole-model accuracy
- the Darknet `.cfg` parser (including `[shortcut]`), the `.weights` reader and batch-norm
  folding
- flash-style sources (`BufferStream`)
- the optimised backends against the reference, parallel ranges and `Task`
- end-to-end detection

`tests/host/second_tu.cpp` is a second translation unit that includes the header. Linking it
with the tests checks that the header-only library has no duplicate symbols.

A strict target compiles the library as C++11 and C++17 with `-fno-exceptions -fno-rtti`, the
flags used by Arduino cores.

## Sanitizers

```bash
cmake -S . -B build-asan -DTINYYOLO_SANITIZE=ON
cmake --build build-asan --target tinyyolo_tests
./build-asan/tinyyolo_tests
```

## Host tools

The desktop C++ tools are in `desktop/` and their CMake file is `desktop/CMakeLists.txt`. The
root `CMakeLists.txt` adds that directory. The binaries are written to `build/desktop/`:

| Target | Source | What it does |
|---|---|---|
| `tinyyolo_make_weights` | `desktop/make_darknet_weights.cpp` | writes a Darknet `.weights` file matching a `.cfg` (pseudo-random values, for tests) |
| `tinyyolo_export_blob` | `desktop/export_model_blob.cpp` | converts a Darknet `.cfg` and `.weights` into the int8 model blob for flash |
| `yolo11_check`, `yolo26_check` | `desktop/yolo/yolo_check.cpp` | every layer of the graph against the PyTorch reference of the export |
| `tinyyolo` | `desktop/tinyyolo/tinyyolo.cpp` | the common desktop tool: a Darknet model or a YOLO export, on the webcam, still pictures or a board stream, with the results in a window (see below) |

Example, after the build:

```bash
build/desktop/tinyyolo_make_weights model.cfg model.weights
build/desktop/tinyyolo --yolo26 models/yolo26n/export --image photo.ppm --no-window --frames 1
```

The YOLO tools include the generated tables from `tools/yolo/` (`yolo11n_graph.inc`,
`yolo26n_graph.inc`), which `tools/yolo/yolo_export.py` writes. Run the export first for a model,
as described in [Models](Models.md#yolo11-ultralytics). The Python scripts are in `tools/`, grouped by
version: `tools/yolo/` (shared), `tools/yolo11/` and `tools/yolo26/` (fetch scripts).

The desktop tools use `DotProductBackend`. On x86-64 the desktop CMake file builds them with `-mavx2`,
and the int8 convolutions then use AVX2 with exact int32 sums; without AVX2 the same code runs as scalar
code, with the same results. An int8 graph run at 320x320 takes about 1 s, and a float32 run about 2 s.
`tinyyolo` runs YOLOv4-tiny (`--model`) in about 10 s (weights-only int8), 7.5 s (float32) and 3.3 s with
`--calibrate`. The times in the [README](../README.md#performance) are the mean of three runs at 320x320
with `--size 320` and `-DCMAKE_BUILD_TYPE=Release`, on a machine with other load.

### The tinyyolo tool

`tinyyolo` is the common desktop tool. It runs any model of the repository on the webcam or on
still pictures, and shows the picture with the detection boxes and the results in a window:

- a Darknet model (`--model <base>`, e.g. `models/yolov4-tiny/yolov4-tiny`), or the built-in sample;
- by default (no model option), the YOLO26n export in `models/yolo26n/export`, the fastest model on the desktop, int8 at 320x320;
- a YOLO11 or YOLO26 export (`--yolo11 <folder>`, `--yolo26 <folder>`, e.g. `models/yolo11n/export`);
- the webcam (`--device <path>`, default `/dev/video0`) or still pictures (`--image <ppm>`, repeated);
- int8 (default) or float32 (`--float`); `--calibrate` for int8 activations of a Darknet model;
  `--per-tensor`, `--size N`, `--score T` and `--names <file>` for the other settings;
- a board stream (`--serial <port>`, `--rotate 0|90|180|270`): the picture and detections of a board running
  `examples/CameraStream` or one of the `*CameraStream` examples, shown in the same window;
- `--frames N` stops after N recognitions, and `--no-window` prints each recognition and its time.

```bash
build/desktop/tinyyolo --yolo11 models/yolo11n/export --size 320
build/desktop/tinyyolo --model models/yolov4-tiny/yolov4-tiny --calibrate --device /dev/video0
build/desktop/tinyyolo --yolo26 models/yolo26n/export --image photo.ppm --no-window --frames 1
```

The window needs SDL2 and the TinyCamera, TinyGPU and TinyMaterialDesign libraries in the folders next
to this one (or set `TINYCAMERA_DIR`, `TINYGPU_DIR` and `TINYMD_DIR`). The window build is off by default:

```bash
cmake -S . -B build -DTINYYOLO_BUILD_CAMERA_VIEW=ON
cmake --build build --target tinyyolo
build/desktop/tinyyolo --serial /dev/ttyACM0 --rotate 180
```

## Desktop platform classes

On desktop, `tinyyolo::Print` and `tinyyolo::Stream` are generic classes with the same member
API as Arduino's (`src/TinyYolo/platform/Io.h`). `Logger::begin()` accepts any of them.
