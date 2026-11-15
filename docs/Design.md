# Design

## Layers

| Layer | Role |
|---|---|
| `Detector` | Facade: `begin()` builds and loads the network, `detect()` runs one frame |
| `Preprocessor`, `ImageView` | Letterbox and bilinear resize into the network input |
| `Model`, `Layer` subclasses | Graph of layers built from a `NetworkSpec`. Shapes are inferred and activation memory is shared between layers |
| `Backend` (strategy) | Compute kernels: `ReferenceBackend`, `DotProductBackend`, `Esp32Backend` |
| `Postprocessor` | YOLO box decoding, per-class greedy NMS, mapping back to the source image |
| `WeightSource` | Parameter sources: arrays, streams, Darknet files |
| `NetworkDescription` | Parses Darknet `.cfg` and `.names` into a `NetworkSpec` |
| `Logger` | Static singleton with levels, writes to any `Print` |

## Errors

Every fallible call returns a `Status`. The function that detects a failure logs the reason
before returning, so a failure always leaves a message in the log.

## Header-only layout

The library has no `.cpp` files. All functions are `inline` or class members. The umbrella
header `src/TinyYolo.h` includes everything. Source files live in `src/TinyYolo/`, grouped
by layer.

## Platform abstraction

- `platform/Io.h`: on Arduino, `tinyyolo::Print` and `tinyyolo::Stream` are aliases of the
  core classes. On other platforms the library provides its own classes with the same API.
- `platform/Features.h`: detects the optional features of the build: FreeRTOS and PSRAM on
  the ESP32 family, esp-dsp when the Arduino core provides it, the RP2040/RP2350 PSRAM allocator
  (`__psram_malloc`, Arduino-Pico), and std::thread on hosts.
- `platform/Platform.h`: `DefaultBackend` is `Esp32Backend` on ESP32, `CmsisBackend` on RP2040 and
  RP2350 (it falls back to the portable kernels when CMSIS-NN is not enabled), and
  `DotProductBackend` elsewhere. `Detector::defaultBackend()` gives access to it, for example to set the
  worker count.
- `platform/Memory.h`: on the ESP32 family every buffer is first requested from internal RAM;
  when that fails it is placed in PSRAM, if the board has PSRAM. On the RP2040/RP2350 the buffers
  come from `malloc()` and `__psram_malloc()` in the same order. `setMemoryPlacement(PsramFirst)`
  reverses the order (used for speed comparisons), and `setPsramAllocation(false)` turns PSRAM
  off.
- `platform/Parallel.h`: `parallelRange()` splits a loop range across workers. On hosts it
  uses `std::thread`; on the ESP32 it uses FreeRTOS tasks, with the extra task on the other
  core. The caller runs one share itself. If a worker task cannot be created (its stack needs
  internal RAM), the caller runs that share too.
- `platform/Task.h`: `Task` runs one function on a thread or FreeRTOS task, with `begin()`,
  `end()`, `isRunning()` and `isEnded()`.
- `platform/KernelOptimize.h`: compiles the kernel headers with `-O2` on the ESP32, although the
  Arduino core builds sketches with `-Os`.
- `platform/Clock.h`: a microsecond clock for `Model::enableProfiling()`, which times each layer
  of a run.

## Model memory

- **Activations:** after the shapes are known, `Model::build()` places every layer output
  in one shared arena. An output lives from the layer that writes it to the last layer that
  reads it. Outputs with non-overlapping lifetimes share memory. YOLO head outputs live until
  the end of the run. `Model::activationBytes()` reports the result, and
  `activationBytesWithoutReuse()` the size without sharing.
- **Weights:** float32 or int8, per output channel (see [Quantization](Quantization.md)).
  Convolutions that feed a YOLO head stay float32 when the model uses int8.
- **Where the weights live:** a model blob is read in place (through the flash mapping) by
  default. Int8 weights can be copied into RAM instead with `ModelBlobSource::setCopyWeights(true)`:
  they are then loaded into PSRAM first. On the ESP32-P4 this is about 3x faster for YOLOv4-tiny
  (see [Quantization](Quantization.md#esp32-p4)).
- **Int8 activations:** `Model::calibrate()` (or `Detector::calibrate()`) runs one float pass and
  sets the scales of the int8-weight convolutions. Those convolutions then exchange int8 data
  inside themselves; the rest of the network stays float32. The int8 input and output buffers
  are shared by all such convolutions.
- **Flash-resident models:** `BufferStream` and `ArrayWeightSource` read the model and
  weights from program memory (`PROGMEM` arrays).

## Weight formats

- TinyYolo's own format: little-endian float32, layer by layer. A conv layer contributes its
  weights (OIHW: `[out][in][kh][kw]`) followed by its bias (`[out]`). Int8 conv weights are
  stored OHWI (`[out][kh][kw][in]`) with one float scale per output channel; see
  [Quantization](Quantization.md).
- Darknet format: see [Models](Models.md). Batch-norm is folded when the weights are read.

## Backends

| Backend | Used on | Convolution | Threads |
|---|---|---|---|
| `ReferenceBackend` | every platform (tests compare against it) | direct loops | none |
| `DotProductBackend` | hosts (default) | patch gathering, one dot product per output channel; int8 convolutions (`conv2dQ`) with AVX2 on x86-64 (`-mavx2`), exact int32 sums | `std::thread`, up to 4 |
| `Esp32Backend` | ESP32 family (default) | as `DotProductBackend`, dot product with esp-dsp | FreeRTOS task on the second core |
| `CmsisBackend` | RP2040 and RP2350 (default) | int8 convolutions with CMSIS-NN (Arm DSP kernels, vendored), one rounding in the requantization; the activation is a lookup table on the int8 output; depthwise and other operations as `ReferenceBackend` | none |
| `Esp32NnBackend` | ESP32-S3 and ESP32-P4 (chosen explicitly) | int8 convolutions: on the S3 the vendored ESP-NN (3x3, depthwise) and our exact kernel for the linear 1x1 layers; on the P4 the vendored esp-dl PIE assembly kernels (3x3, 1x1, depthwise), ESP-NN for other shapes; other operations as `Esp32Backend` | FreeRTOS task on the second core |

All backends implement the same operations: convolution (float32 and int8 weights, and int8
activations through `conv2dQ()`), max pooling (float and int8), upsampling, concatenation, copy
and the residual add.

To add a backend, derive from `DotProductBackend` and override `dot()` (float32 weights)
or `dotInt8()` (int8 weights). For operations that need a different algorithm, override the
operation itself, as `Esp32NnBackend` does for `conv2dQ()`. ESP-NN and its build flag are
described in [Quantization](Quantization.md).

## Limitations

- The examples are compile-checked for ESP32, ESP32-S3, SAMD (MKR WiFi 1010) and RP2040
  (Pico). The RP2350B (SpotPear RP2350B-MINI-A, 8 MB PSRAM) runs the YOLO11n and YOLO26n graphs
  from test sketches that are not in the repository yet. The test suite runs on Linux. The streaming example (`Yolov4CameraStream`) has run on an
  ESP32-S3-EYE, with YOLOv4-tiny at 192x192 (see the example's README for the timings). The other
  examples have not been run on a board.
- Without an optimised kernel (ESP-NN on the S3, the assembly kernels on the P4, CMSIS-NN on the
  RP2350B), the int8 dot product is scalar code, so int8 weights save memory but not time.
- Buffers go to internal RAM first and to PSRAM when internal RAM cannot hold them. The frame
  times of every board are in the [Performance](Performance.md).
- AVR boards, TPU, GPU and FPGA accelerators are not supported.
