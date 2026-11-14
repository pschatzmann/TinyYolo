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
| `yolo11_detect`, `yolo26_detect` | `desktop/yolo/yolo_detect.cpp` | detections on a PPM image, through the same `GraphYolo11` class as the boards. `--size N` sets the square network input (640 by default, 320 for the README timings), `--float` runs float32 weights |
| `yolo11_check`, `yolo26_check` | `desktop/yolo/yolo_check.cpp` | every layer of the graph against the PyTorch reference of the export |
| `darknet_detect` | `desktop/darknet/darknet_detect.cpp` | a Darknet model (`.cfg`, `.names`, `.weights`) on a PPM image, through `TinyYoloModel` as on the boards; `--size N` checks the cfg input size, `--runs N` sets the number of timed runs, `--float` uses float32 weights, `--calibrate` also quantizes the activations (calibrated on the image) |
| `camera_view` | `desktop/camera_view/camera_view.cpp` | the camera viewer (see below) |

Example, after the build:

```bash
build/desktop/tinyyolo_make_weights model.cfg model.weights
build/desktop/yolo26_detect models/yolo26n/export photo.ppm
```

The YOLO tools include the generated tables from `tools/yolo/` (`yolo11n_graph.inc`,
`yolo26n_graph.inc`), which `tools/yolo/yolo_export.py` writes. Run the export first for a model,
as described in [Models](Models.md#yolo11-ultralytics). The Python scripts are in `tools/`, grouped by
version: `tools/yolo/` (shared), `tools/yolo11/` and `tools/yolo26/` (fetch scripts).

The desktop tools use `DotProductBackend`. On x86-64 the desktop CMake file builds them with `-mavx2`,
and the int8 convolutions then use AVX2 with exact int32 sums; without AVX2 the same code runs as scalar
code, with the same results. An int8 graph run at 320x320 takes about 1 s, and a float32 run about 2 s.
The Darknet tool runs YOLOv4-tiny in about 10 s (weights-only int8), 7.5 s (float32) and 3.3 s with
`--calibrate`. The times in the [README](../README.md#performance) are the mean of three runs at 320x320
with `--size 320` and `-DCMAKE_BUILD_TYPE=Release`, on a machine with other load.

### Camera viewer

`camera_view` shows a desktop webcam with the detections of a board or a Darknet model. It needs
SDL2 and the TinyCamera, TinyGPU and TinyMaterialDesign libraries in the folders next to this one
(or set `TINYCAMERA_DIR`, `TINYGPU_DIR` and `TINYMD_DIR`). It is off by default:

```bash
cmake -S . -B build -DTINYYOLO_BUILD_CAMERA_VIEW=ON
cmake --build build --target camera_view
build/desktop/camera_view --serial /dev/ttyACM0 --rotate 180
```

## Desktop platform classes

On desktop, `tinyyolo::Print` and `tinyyolo::Stream` are generic classes with the same member
API as Arduino's (`src/TinyYolo/platform/Io.h`). `Logger::begin()` accepts any of them.
