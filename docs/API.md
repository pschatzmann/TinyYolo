# API

Everything is in the `tinyyolo` namespace. On Arduino, `#include <TinyYolo.h>` brings the
namespace into scope. On other platforms, write `tinyyolo::` or add a using-directive.

## Detector

```cpp
Detector detector;
DetectorConfig config;                  // score_threshold 0.25, iou_threshold 0.45, max_candidates 1024
config.score_threshold = 0.3f;
config.weight_precision = WeightPrecision::Float32;  // or Int8, see Quantization.md
detector.begin(network_spec, weights, config);   // builds the network, loads the weights

Detections detections;
detections.reserve(20);                 // maximum number of results
detector.detect(image, detections);     // one frame
detector.label(detections[0].class_id); // class name, "?" if there is none
```

- `begin(spec, weights, config, backend)` builds the network and may be called again to
  change it. `backend` is optional. The default is the platform's `DefaultBackend`.
- `defaultBackend()` gives access to the built-in backend, for example to set the worker
  count before `begin()`.
- `detect(image, out)` letterboxes the image into the network input, runs the network,
  decodes the YOLO heads, applies per-class NMS and writes boxes in source-image pixels.
- `calibrate(image)` chooses the int8 activation scales from one representative image. Call it
  after `begin()` when `weight_precision` is `Int8` and before the first `detect()`. See
  [Quantization](Quantization.md).
- `enableProfiling(true)` times every layer of the following `detect()` calls. `layerMicros(i)`
  returns the time of layer `i` in microseconds, and `model().layerCount()` the number of layers.

```cpp
static Esp32NnBackend backend;           // ESP32-S3: int8 convolutions with ESP-NN
detector.begin(network.spec(), weights, config, &backend);
detector.calibrate(camera_image);        // int8 activations, once
```

## Detection and Detections

```cpp
struct Detection { float x, y, width, height; float score; int class_id; };
```

`Detections` is a fixed-capacity list. `reserve(n)` allocates it once, and `push` reports
overflow instead of growing. It can be used in a range-for loop.

## ImageView and PixelFormat

```cpp
ImageView image(data, width, height, PixelFormat::RGB888, row_bytes /* optional */);
```

Formats: `Gray8`, `RGB888`, `BGR888`, `RGB565Big` (high byte first) and `RGB565Little`.
The image is not copied, so the buffer must stay valid during `detect()`.

## Drawing

`drawRectangle()` and `drawDetection()` (in `image/Draw.h`) outline boxes on an RGB888
buffer. Pixels outside the buffer are skipped.

```cpp
drawDetection(frame, width, height, detection, Rgb{255, 0, 0}, 1);
```

## Logger

```cpp
TinyYoloLogger.begin(Serial, TinyYoloLogLevel::Info);   // any Print
TinyYoloLogger.setLevel(TinyYoloLogLevel::Warning);
TINYYOLO_LOGE("Module", "value %d", v);          // also LOGW, LOGI, LOGD
```

Levels: `Debug`, `Info`, `Warning`, `Error`, `Off`. Nothing is printed until `begin()`.
Every error and warning in the library is logged. Output format: `[E] Module: message`.
`TINYYOLO_MIN_LOG_LEVEL` (0 to 4) removes messages at compile time, and
`TINYYOLO_LOG_BUFFER_SIZE` sets the line buffer size.

## Status

Fallible functions return `Status`: `Ok`, `InvalidArgument`, `InvalidModel`,
`ShapeMismatch`, `OutOfMemory`, `WeightError`, `NotInitialized` or `Unsupported`.
`statusName(status)` returns the name as text.

## Network description

`NetworkSpec` is the plain-data description used by `Detector::begin()`. Build it in code
with the `LayerSpec` factories, or load it from a Darknet file (see [Models](Models.md)).

```cpp
static const float kAnchors[] = {8, 8, 16, 16, 32, 32};
static const LayerSpec kLayers[] = {
    LayerSpec::conv(16, 3, 2),                     // conv(filters, kernel, stride, activation, pad, batch_norm)
    LayerSpec::maxPool(2, 2),                      // maxPool(kernel, stride, total_padding)
    LayerSpec::upsample(2),
    LayerSpec::route(0, 2),                        // concatenate two layers (absolute indices)
    LayerSpec::add(2, 0),                          // residual add of two layers (same shape)
    LayerSpec::conv(24, 1, 1, Activation::Linear), // conv feeding the head
    LayerSpec::yolo(3, kAnchors, 3),               // yolo(classes, anchors, count, decode, scale_xy)
};
static const NetworkSpec kNetwork = {64, 64, 3, kLayers, 7, nullptr, 0};
```

- `LayerSpec::input(i)` makes a layer read from layer `i` instead of the previous layer.
- `LayerSpec::withPrecision(p)` sets the weight precision of one convolution (see
  [Quantization](Quantization.md)).

## Model

`Model` builds and runs a network without the detection pipeline:

```cpp
Model model;
model.build(spec, backend, WeightPrecision::Float32);  // reuse_activations = true by default
model.loadWeights(source);
model.input().fill(0.5f);     // fill the network input
model.run();
model.head(0).output();       // raw YOLO head output (HWC)
```

`activationBytes()` and `weightBytes()` report the memory used after `build()`.

Int8 activations (see [Quantization](Quantization.md)):

```cpp
model.calibrate(sample);            // sample: a Tensor with the network input shape
model.int8ActivationLayers();       // number of convolutions that now use int8 activations
```

Profiling:

```cpp
model.enableProfiling(true);
model.run();
model.layerMicros(i);               // time of layer i in the last run, in microseconds
```

Buffer placement (ESP32): every buffer is requested from internal RAM first, then from PSRAM.
Include `TinyYolo/platform/Memory.h` for:

```cpp
tinyyolo::setMemoryPlacement(tinyyolo::MemoryPlacement::PsramFirst);  // reverse the order
tinyyolo::setPsramAllocation(false);                                  // no PSRAM at all
```

A single buffer can name its own placement: `Buffer<T>::allocate(count, MemoryPlacement::PsramFirst)`.
The int8 weights of the convolutions use this.

Weights in RAM (blob models): by default the int8 weights of a blob are read in place, through the
flash mapping. To copy them into RAM when the model loads:

```cpp
tinyyolo::ModelBlobSource weights(blob, blob_size);
weights.setCopyWeights(true);   // before the model loads its weights
```

See [Quantization](Quantization.md#esp32-p4) for when this matters.

## Backends

| Class | Use |
|---|---|
| `ReferenceBackend` | portable, the correctness baseline |
| `DotProductBackend` | hosts (default), threads; int8 convolutions with AVX2 on x86-64 |
| `Esp32Backend` | ESP32 family (default), esp-dsp dot products |
| `CmsisBackend` | RP2040 and RP2350 (default): int8 convolutions with CMSIS-NN. Needs the Arm DSP extension and GCC 12 or newer; otherwise the portable kernels are used. |
| `Esp32NnBackend` | ESP32-S3 and ESP32-P4 (chosen explicitly): int8 convolutions with the vendored ESP-NN (S3) or the P4 assembly kernels. Pass it to `Detector::begin()` or `Model::build()`. See [Quantization](Quantization.md#esp-nn-on-the-esp32-s3). |

`Detector` uses the `DefaultBackend` of the platform (see `platform/Platform.h`) when it is given none.

## High-level classes

These classes wrap the calls above for the common cases. They are included by `TinyYolo.h`.

### Model sources

`ModelSource` gives the three Darknet streams of one model. `FlashModelSource` reads arrays
from program memory; `SdModelSource` reads `<base>.cfg`, `<base>.names` and `<base>.weights`
from the SD card.

```cpp
#include <TinyYolo.h>
#include "DarknetModel.h"                       // from tools/bin2header.py
FlashModelSource source(kModelCfg, kModelNames, kModelWeights);
```

```cpp
#include <TinyYolo.h>
#include <TinyYolo/app/SdModelSource.h>         // needs the SD library
SdModelSource source("/model", SS);             // base path, chip-select pin
```

`SdModelSource` needs the SD library. Without it, the include stops with an error that names
the missing library.

### TinyYoloModel

Owns the network description, the weights, the detector and the results:

```cpp
TinyYoloModel model;
model.begin(source);                  // loads the model; returns Status
model.detect(image);                  // one frame
model.detections();                   // results of the last detect()
model.detector();                     // labels, configuration
model.inputWidth();                   // network input size
```

`begin()` accepts a `DetectorConfig` and the capacity of the results list (default 20).

### SyntheticFrame

Generates test frames without a camera: a gradient with a bright square that moves on every
call.

```cpp
SyntheticFrame frames(64, 64);
ImageView image = frames.next();
```

### Output

`Output` prints formatted text, one printf-style call per line. It prints to `Serial` on
Arduino; `Output(Print&)` prints to any other target. A sketch defines one `Output out;` and
prints a detection as one line per box:

```cpp
Output out;                                   // global: prints to Serial
for (const Detection& d : model.detections()) {
  out.printf("%s %.2f at %d,%d %dx%d", model.detector().label(d.class_id), d.score,
             (int)d.x, (int)d.y, (int)d.width, (int)d.height);
}
```

### CameraFrame (TinyCamera)

Captures frames with TinyCamera and scales them to the network input (RGB565). Include
`<TinyYolo/TinyCamera/CameraFrame.h>` after `<TinyYolo.h>`. It requires the TinyCamera library;
without it, the include stops with an error.

```cpp
CameraFrame camera_frame(64, 64);
camera_frame.begin(config);           // camera_config_t from TinyCamera, with the board's pins
if (camera_frame.next()) model.detect(camera_frame.image());
```

### LcdOverlay (TinyGPU)

Shows a frame and its detection boxes on a TinyGPU LCD board. Include
`<TinyYolo/TinyGPU/LcdOverlay.h>` after `<TinyYolo.h>`. It requires the TinyGPU library.

```cpp
LCDBoardGuitionESP32_LVGL_2_4Display board;
LcdOverlay overlay(board);
overlay.begin();
overlay.show(image, model.detections());
```

## Weights

Weights are float32 by default. With `WeightPrecision::Int8` they are quantized per output
channel when the model loads: see [Quantization](Quantization.md).

| Source | Use |
|---|---|
| `ArrayWeightSource(float*, count)` | Parameters in a RAM array, in TinyYolo's own order |
| `StreamWeightSource(stream)` | Little-endian float32 from any `Stream`, in TinyYolo's own order |
| `DarknetWeightSource(stream)` | Darknet `.weights` file, with batch-norm folded in |
| `BufferStream(data, size)` | Read-only `Stream` over a byte array (flash or RAM) |

Custom sources derive from `WeightSource` and implement `read()`. For a whole convolution,
`readConv()` can be overridden instead.
