# Quantization

Convolution weights can be stored as int8 instead of float32. This cuts the weight memory by
about 3.7x on the sample model.

## How it works

Weights-only, symmetric, per output channel:

- For each output channel `o`, `scale[o] = max|w[o]| / 127`.
- Each weight becomes `q = round(w / scale[o])`, clamped to [-127, 127].
- At inference a convolution accumulates `sum(x * q)` in float, multiplies by `scale[o]` and
  adds the float bias. Activations stay float32.

Only the weights are rounded. Biases and activations stay float32.

Batch-norm is folded into the float weights and bias before quantization, so Darknet models
quantize correctly.

## Using it

```cpp
DetectorConfig config;
config.weight_precision = WeightPrecision::Int8;   // default: WeightPrecision::Float32
detector.begin(network.spec(), source, config);
```

Or, with a `Model` directly:

```cpp
Model model;
model.build(spec, backend, WeightPrecision::Int8);
```

`model.weightBytes()` reports the memory used for parameters after `build()`.

## Which layers are int8

With `WeightPrecision::Int8`, each convolution is chosen in this order:

1. **Per-layer override:** if the layer has one (see below), it is used.
2. **YOLO head rule:** a convolution whose output feeds a `[yolo]` head stays float32. The
   head's outputs are decoded into boxes and scores, so these layers are the most sensitive
   to rounding. They are also small: a 1x1 convolution in the sample model.
3. **Model precision:** every other convolution uses int8.

The head rule applies only to int8 models. A float32 model stays float32 everywhere.

### Per-layer override

```cpp
static const LayerSpec kLayers[] = {
    LayerSpec::conv(16, 3, 2),                                   // int8 (model default)
    LayerSpec::conv(24, 1, 1, Activation::Linear)                // feeds the head: float32
        .withPrecision(WeightPrecision::Int8),                   // ...forced to int8 here
    LayerSpec::yolo(3, kAnchors, 3),
};
```

`withPrecision()` works in both directions: a float32 layer in an int8 model, or an int8
layer in a float32 model. It applies only to convolutions.

### Logging

At Debug level, the model logs each convolution's precision when it is built:

```cpp
TinyYoloLogger.begin(Serial, TinyYoloLogLevel::Debug);
```

`model.layer(i).weightBytes()` gives the memory of one layer after `build()`.

## Results

Sample model with 6,824 parameters (not in the repository; see [Models](Models.md#sample-model)):

| | Float32 | Int8 |
|---|---:|---:|
| Weight memory | 27,296 bytes | 7,384 bytes (3.7x smaller) |
| Worst head output error | - | 0.18% of the output range |

The error was measured with pseudo-random weights over five random inputs. A trained model
can differ, so check the detections on your own model.

## Int8 activations

Int8 weights can be combined with int8 activations. A convolution then quantizes its input,
runs the integer dot products and hands its output on as float32:

```cpp
detector.begin(network.spec(), source, config);   // config.weight_precision = Int8
detector.calibrate(calibrationImage);             // one representative image
detector.detect(frame, detections);
```

`calibrate()` runs one float pass and records the largest value of every layer output. Each
int8 convolution then gets an input scale (from its input) and an output scale (from its
own output), both `max / 127`. Calibrate with an image like the ones you will detect: values
far above the calibration range are clipped.

Only convolutions with int8 weights use int8 activations. Convolutions that feed a YOLO head
stay float32 (see above), and so do the routes, shortcuts, upsampling and the head itself.
`Model::int8ActivationLayers()` reports how many convolutions switched.

### Results

YOLOv4-tiny at 416 pixels, one 375x250 photo (a horse, COCO class 17):

| | Detections | Box (x, y, w, h) |
|---|---|---|
| Float32 | 1 (score 0.96) | 92, 74, 83, 133 |
| Int8 weights | 1 (score 0.96) | 92, 74, 83, 133 |
| Int8 weights + activations | 1 (score 0.96) | 91, 77, 84, 128 |

On the raw head outputs, the mean error is about 1.6% of the output range with int8
activations, against about 0.5% with int8 weights alone. The largest single error is about
11-16% of the range. These are measurements on one photo, not a benchmark.

### Speed

On the desktop the int8 graph uses AVX2 (x86-64), and runs about as fast as the float32 path (1.0 s and 2.2 s for
YOLO11n at 320x320; see the README). On the ESP32-S3 the speed comes from the ESP-NN
kernel (see below).

### Limits

- Calibration uses the largest values of one run. Inputs far above that range are clipped.
  Clipping the large negative values of a Leaky convolution was tested as a fix: covering the
  full negative range made the quantization step about ten times coarser and the error grew
  (mean head error 0.28 to 1.8), so the range is taken from the activated output as before.
- Only convolutions with int8 weights use int8 activations; routes, shortcuts, upsampling,
  residual adds, and the YOLO heads stay float32.

## ESP-NN on the ESP32-S3

`Esp32NnBackend` (`backend/Esp32NnBackend.h`) runs the int8 convolutions with the ESP-NN kernels
of the Arduino core. It is used on the ESP32-S3 (and on the P4 when the P4 assembly kernels are
switched off). Other layers and float convolutions are inherited from `Esp32Backend`.

```cpp
static Esp32NnBackend backend;                     // instead of the default backend
Model model;
model.build(spec, backend, WeightPrecision::Int8);
```

### Build

The ESP-NN kernels are part of the library (`backend/esp32s3/esp-nn/`, ESP-NN v1.5.0, Apache-2.0),
as headers that are compiled with the sketch. No include path or link flag is needed.

```sh
arduino-cli compile --fqbn esp32:esp32:esp32s3:PSRAM=opi,FlashSize=16M,PartitionScheme=custom <sketch>
```

On the S3 the convolutions use this path:

- **3x3 and other spatial convolutions:** the S3-specific ESP-NN kernel (`esp_nn_conv_s8_esp32s3`). It is
  correct for 4 and for multiples of 16 input channels. For other input channel counts, and for output
  channels that are not multiples of 16, the layer is padded with zero channels (`espNnConvPadded`) and the
  padding output channels are dropped.
- **Depthwise convolutions:** the generic `esp_nn_depthwise_conv_s8_opt` kernel. The S3-specific depthwise
  kernel is wrong.
- **Linear 1x1 convolutions:** our own exact kernel (`exactConv2dQ` in `Esp32NnBackend.h`). It sums the
  products exactly in int32, using the S3 vector dot product (`esp_nn_dot_s8_unaligned_esp32s3`) for
  interior pixels. It is used for the detection heads and the neck projections, which determine the final
  scores; the ESP-NN pointwise path gives different scores for them on the YOLO11n graph, though it matches
  the reference on random data.
- **Requantization:** `prepareRequantSingle` writes the multiplier as `ratio * 2^31` with shift 0, so the
  kernel rounds once. The normalized form (`prepareRequant`, a power-of-two shift) rounds twice and is
  kept for the P4 paths only.
- **Activated layers (SiLU, ...):** ESP-NN applies the activation after the rounding, so these layers
  differ from the reference by one step in about 2 % of values. Making them exact with our own kernel
  took about 17 s for the graph on the S3, so they stay on ESP-NN.

### Layout and requantization

- Int8 convolution weights are stored **OHWI** (`[out][kh][kw][in]`), the layout ESP-NN reads.
  Float weights stay OIHW.
- Requantization uses the TFLite convention: for output channel `o`, the real multiplier
  `in_scale * w_scale[o] / out_scale` is stored as `mult * 2^(shift - 31)`. The bias is converted
  to the int32 accumulator scale.
- ESP-NN only clamps to the int8 range. Activations other than Linear run as a float pass over
  the int8 output.

### Memory

- The int8 input and output of a convolution are shared by all int8 convolutions of the model:
  one buffer each, sized for the largest layer. A convolution uses them only during its own
  forward pass.
- ESP-NN's scratch buffer grows with the rows it convolves: 2.4 MB for a 96x96 layer. The
  backend therefore computes the output in bands of rows, choosing the largest band whose scratch
  fits 512 KB. The input rows of a band are copied with zero rows at the picture edges.

### 1x1 convolutions

On the S3 the linear 1x1 layers (the detection heads and the neck projections) use our own exact kernel
(`exactConv2dQ`), not ESP-NN. The ESP-NN pointwise path (`esp_nn_fully_connected_per_ch_s8`) matches the
reference on random data, but on the YOLO11n graph it changes the final scores, so the graph does not use
it. The exact kernel sums the products in int32 and rounds once. Activated 1x1 layers stay on ESP-NN.

### Channel counts that are not multiples of 4

ESP-NN's 3x3 kernel needs input channels in multiples of 4. On the int8 path, the first
YOLOv4-tiny layer (3 input channels) is padded with zero channels, and its int8 weights get
matching zero columns (`ConvLayer::enableInt8Activations()`). The float-activation path keeps the
unpadded weights.

### Memory placement

Every buffer is requested from internal RAM first and falls back to PSRAM when that fails
(`platform/Memory.h`). `setMemoryPlacement(MemoryPlacement::PsramFirst)` reverses the order.

### Head-feeding layers

Only the two convolutions that directly feed a YOLO layer stay float32 with int8 weights (see
"Which layers are int8"): layers 29 and 36, both 1x1 with 255 outputs. Together they take 1.3 s
per frame. They use the float dot product, which has no ESP-NN path.

Converting them to int8 was checked on the desktop with the YOLOv4-tiny weights and int8
activations. The raw head outputs (range about 15-18) change by 0.15-0.20 on average (0.8-1.3% of
the range). The largest single change is 1.7-3.2 (11-18% of the range).

The int8 heads are selected with `-DTINYYOLO_INT8_HEADS` (the streaming example). On the test photo
the detection is the same (horse, class 17), but the box width changes by about 8%. The default stays
float32.

### Not supported

- **Channel counts that are not multiples of 4 on the 3x3 kernel.** Padded on the int8 path (see
  above); other paths use the portable kernel.
- **Strided or padded 1x1 kernels.** The portable kernel.

## CMSIS-NN on the RP2040 and RP2350

`CmsisBackend` (`backend/cmsis/CmsisBackend.h`) runs the int8 convolutions with CMSIS-NN, the Arm
kernels for the DSP extension. The kernels are vendored as headers (`backend/cmsis/`, Apache-2.0, see
`LICENSE` there); only the convolution kernels and their support functions are included. They are used
on the Cortex-M33 of the RP2350 when the compiler is GCC 12 or newer (`arm_cmsis_nn_config.h`). The
Cortex-M0+ of the RP2040 and any other core use the portable kernels.

- **Requantization:** the per-channel multiplier is `ratio * 2^31` with shift 0, so the kernel rounds
  once, as the reference does. The normalized form (shift below zero) rounds twice.
- **Activations:** the convolution output is linear, and SiLU (and the other activations) are applied
  as a lookup table on the int8 output. The layers with an activation therefore differ from the
  reference by one step on up to about 4 % of the values.
- **Depthwise convolutions and the other operations** use the portable `ReferenceBackend`.

The RP2350B runs at 300 MHz with `-Ofast` and PSRAM at 133 MHz (the Arduino-Pico board options
`freq=300`, `opt=Fast`, `psramfreq=freq133`). The frame times are in the
[README](../README.md#performance).

## ESP32-P4

The ESP32-P4 (Waveshare ESP32-P4-WIFI6, 32 MB flash, PSRAM) uses the same `Esp32NnBackend`. Int8
3x3 and 1x1 convolutions whose channel counts are multiples of 16 run on the esp-dl PIE assembly
kernels, vendored in `backend/esp32p4/` (MIT, see `LICENSE.esp-dl.txt`). Depthwise 3x3 convolutions
run there too when all channels share one power-of-two scale. Other shapes fall back to the ESP-NN
RISC-V PIE kernels (`esp_nn_riscv_pie.h`), which have the same API as the S3 kernels. The build needs `ChipVariant=postv3` (the default "Before v3.00" build boot-loops on this
board) and the P4 ESP-NN library:
`-L <esp32p4-libs>/lib -lespressif__esp-nn`. The check cases match the S3 (max difference 1 LSB).

The model is in the flash partition `model` (see `examples/Yolov4CameraStream/partitions.csv` for
the layout). The P4 flash cache maps only the first 16 MB of flash, so the model partition must
start below 16 MB and be mapped in pieces that stay in that window.

### Assembly kernels on the P4

- The kernels read 16 input channels per pixel, so a layer runs on them only when its input and output
  channels are multiples of 16. The first YOLO11 convolution has 3 input channels: the graph pads its
  input and weights to 16 channels (zeros) and writes the zero border itself, so the backend needs no
  padded copy. The vendored unaligned kernels for fewer than 16 channels are not used: their output
  did not match the reference.
- The kernels truncate the shifted accumulator. `Esp32NnBackend` adds half of the divisor to the bias,
  which rounds to nearest as ESP-NN does.
- Weights are packed as `(N/16)HWC16`, one 16-byte row per input channel and filter tap.
- A packed-weight cache (`Esp32NnBackend`, 96 entries) keeps the packed filters between frames.

### Weights in RAM

Read in place, the int8 filters come from the flash cache, and the large 3x3 layers are slow
(3 s each for the 128 to 512 channel layers). `ModelBlobSource::setCopyWeights(true)` copies the
int8 weights, scales and biases into PSRAM when the model loads. The layers then read them from
RAM. The copy takes about 6.6 MB of PSRAM for YOLOv4-tiny, which the P4 has and the S3 does not
(the S3 keeps the in-place views).

## Speed

The int8 weight path reads a quarter of the weight bytes, which helps where memory bandwidth is
the limit. The ESP32 weights-only int8 dot product is scalar code: esp-dsp has no mixed
int8-by-float dot product. On the ESP32 with `Esp32Backend`, int8 weights save memory but not
time. Use `Esp32NnBackend` with int8 activations for the faster path.

## Code

- `core/Quantize.h`: `WeightPrecision` and `quantizeRows()`.
- `model/Layer.h`: `ConvLayer` quantizes at load time. The float weights are read into a
  temporary buffer, gathered into OHWI order per output channel, quantized, and the buffer is freed.
- `backend/Backend.h`: `conv2dInt8()`, implemented by `ReferenceBackend` and
  `DotProductBackend` (with the virtual `dotInt8()` hook for platform backends), and `CmsisBackend` (RP2350).
- `backend/cmsis/CmsisBackend.h`: CMSIS-NN `conv2dQ()` on the RP2040 and RP2350 (see the section above).
- `backend/Esp32NnBackend.h`: ESP-NN `conv2dQ()` on the ESP32-S3: the S3-specific 3x3 kernel (in row
  bands), padded where the kernel is not reliable (`espNnConvPadded`), `esp_nn_depthwise_conv_s8_opt`
  for the depthwise layers, and our exact kernel (`exactConv2dQ`) for the linear 1x1 layers. Falls back
  to the portable kernel.
- `backend/esp32s3/esp-nn/`: the vendored ESP-NN sources as headers, collected by `esp_nn_s3_sources.h`.
- `platform/Memory.h`: buffer placement (internal RAM first, PSRAM as fallback).
- `platform/KernelOptimize.h`: the `-O2` pragma for the kernel headers on the ESP32.
- `platform/Clock.h`: the microsecond clock used by `Model::enableProfiling()`.
- `core/QTensor.h`: the int8 tensor with its scale, and `quantizeTensor()` /
  `dequantizeTensor()`.
- `backend/Backend.h`: `conv2dQ()` and `maxPoolQ()` (int8 activations). The defaults are the
  portable reference kernels; a platform backend overrides `conv2dQ()` for speed.
- `model/Layer.h`: `ConvLayer::enableInt8Activations()` and the quantized forward path.
- `model/Model.h`: `calibrate()`. `Detector::calibrate()` runs the same step on an image.
