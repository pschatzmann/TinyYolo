/// Automated desktop tests for TinyYolo. Exit code 0 means every check passed.

#include <math.h>
#include <stdio.h>
#include <string.h>

#include <atomic>
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "TinyYolo.h"
#include "TinyYolo/backend/Esp32NnBackend.h"
#include "TinyYolo/backend/esp32p4/ConvSchedule.h"
#include "TinyYolo/graph/AttentionKernel.h"
#include "TinyYolo/model/ModelBlob.h"

using namespace tinyyolo;

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond)                                                              \
  do {                                                                           \
    ++g_checks;                                                                  \
    if (!(cond)) {                                                               \
      ++g_failures;                                                              \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);            \
    }                                                                            \
  } while (0)

#define CHECK_NEAR(a, b, eps) CHECK(fabs((double)(a) - (double)(b)) <= (eps))

namespace {

class CapturePrint : public Print {
 public:
  size_t write(uint8_t c) override {
    text += (char)c;
    return 1;
  }
  std::string text;
};

class MemoryStream : public Stream {
 public:
  MemoryStream(const void* data, size_t size) : data_((const uint8_t*)data), size_(size) {}
  size_t write(uint8_t) override { return 0; }
  size_t readBytes(char* buffer, size_t length) override {
    size_t n = 0;
    while (n < length && pos_ < size_) buffer[n++] = (char)data_[pos_++];
    return n;
  }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t pos_ = 0;
};

bool contains(const std::string& text, const char* needle) {
  return text.find(needle) != std::string::npos;
}

/// Logger that captures output for the duration of one test.
struct ScopedLog {
  explicit ScopedLog(TinyYoloLogLevel level) { Logger::instance().begin(out, level); }
  ~ScopedLog() { Logger::instance().end(); }
  CapturePrint out;
};

// ---------------------------------------------------------------- Logger

void testLogger() {
  CHECK(&Logger::instance() == &Logger::instance());
  CHECK(&TinyYoloLogger == &Logger::instance());
  {
    ScopedLog log(TinyYoloLogLevel::Warning);
    TINYYOLO_LOGI("Test", "hidden %d", 1);
    CHECK(log.out.text.empty());
    TINYYOLO_LOGW("Test", "warn %s %d", "x", 2);
    CHECK(log.out.text == "[W] Test: warn x 2\r\n");
    TINYYOLO_LOGE("Test", "err %.2f", 1.5);
    CHECK(contains(log.out.text, "[E] Test: err 1.50\r\n"));
  }
  {
    ScopedLog log(TinyYoloLogLevel::Off);
    TINYYOLO_LOGE("Test", "silenced");
    CHECK(log.out.text.empty());
  }
  {
    // No output configured: logging must be a silent no-op.
    Logger::instance().end();
    TINYYOLO_LOGE("Test", "nowhere");
    CHECK(!Logger::instance().isEnabled(TinyYoloLogLevel::Error));
  }
  {
    ScopedLog log(TinyYoloLogLevel::Debug);
    TINYYOLO_LOGD("Test", "debug line");
    CHECK(contains(log.out.text, "[D] Test: debug line"));
    Logger::instance().setLevel(TinyYoloLogLevel::Error);
    CHECK(Logger::instance().level() == TinyYoloLogLevel::Error);
    CHECK(!Logger::instance().isEnabled(TinyYoloLogLevel::Warning));
  }
}

// ---------------------------------------------------------------- Core types

void testStatusAndTensor() {
  CHECK(strcmp(statusName(Status::Ok), "Ok") == 0);
  CHECK(strcmp(statusName(Status::WeightError), "WeightError") == 0);

  Tensor t;
  CHECK(t.allocate(Shape(2, 3, 4)) == Status::Ok);
  CHECK(t.size() == 24);
  CHECK(t.shape() == Shape(2, 3, 4));
  CHECK(t.at(0, 0, 0) == 0.0f);
  t.at(1, 2, 3) = 7.0f;
  CHECK(t.data()[(1 * 3 + 2) * 4 + 3] == 7.0f);  // HWC layout

  Tensor bad;
  {
    ScopedLog log(TinyYoloLogLevel::Error);
    CHECK(bad.allocate(Shape(0, 3, 4)) == Status::InvalidArgument);
    CHECK(contains(log.out.text, "[E] Tensor: invalid shape 0x3x4"));
  }
}

void testActivations() {
  CHECK_NEAR(activate(Activation::Linear, -2.0f), -2.0f, 1e-6);
  CHECK_NEAR(activate(Activation::ReLU, -2.0f), 0.0f, 1e-6);
  CHECK_NEAR(activate(Activation::ReLU, 2.0f), 2.0f, 1e-6);
  CHECK_NEAR(activate(Activation::Leaky, -2.0f), -0.2f, 1e-6);
  CHECK_NEAR(activate(Activation::SiLU, 0.0f), 0.0f, 1e-6);
  CHECK_NEAR(sigmoid(0.0f), 0.5f, 1e-6);
  CHECK(windowOutputSize(8, 3, 2, 1) == 4);
  CHECK(windowOutputSize(2, 3, 1, 0) == -1);
}

// ---------------------------------------------------------------- Reference kernels

void testConvolution() {
  ReferenceBackend be;

  // 3x3 all-ones kernel, pad 1, bias 0.5 over an all-ones 3x3 image.
  Tensor in, out;
  in.allocate(Shape(3, 3, 1));
  in.fill(1.0f);
  out.allocate(Shape(3, 3, 1));
  std::vector<float> w(9, 1.0f);
  float bias = 0.5f;
  ConvParams p = {3, 1, 1, 1, Activation::Linear};
  CHECK(be.conv2d(in, w.data(), &bias, p, out) == Status::Ok);
  CHECK_NEAR(out.at(1, 1, 0), 9.5f, 1e-5);  // full 3x3 window
  CHECK_NEAR(out.at(0, 0, 0), 4.5f, 1e-5);  // corner: 2x2 valid
  CHECK_NEAR(out.at(0, 1, 0), 6.5f, 1e-5);  // edge: 2x3 valid

  // Stride 2, no padding, 2x2 kernel of ones over a counting image.
  Tensor in2, out2;
  in2.allocate(Shape(4, 4, 1));
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) in2.at(y, x, 0) = (float)(y * 4 + x);
  out2.allocate(Shape(2, 2, 1));
  std::vector<float> w2(4, 1.0f);
  float zero = 0.0f;
  ConvParams p2 = {2, 2, 0, 1, Activation::Linear};
  CHECK(be.conv2d(in2, w2.data(), &zero, p2, out2) == Status::Ok);
  CHECK_NEAR(out2.at(0, 0, 0), 10.0f, 1e-5);
  CHECK_NEAR(out2.at(1, 1, 0), 50.0f, 1e-5);

  // Two input channels, two output channels: verifies OIHW weight order.
  Tensor in3, out3;
  in3.allocate(Shape(1, 1, 2));
  in3.at(0, 0, 0) = 2.0f;
  in3.at(0, 0, 1) = 3.0f;
  out3.allocate(Shape(1, 1, 2));
  std::vector<float> w3 = {1.0f, 10.0f,   // out 0: in0, in1
                           0.0f, 1.0f};   // out 1
  float b3[2] = {0.0f, 1.0f};
  ConvParams p3 = {1, 1, 0, 2, Activation::Linear};
  CHECK(be.conv2d(in3, w3.data(), b3, p3, out3) == Status::Ok);
  CHECK_NEAR(out3.at(0, 0, 0), 32.0f, 1e-5);
  CHECK_NEAR(out3.at(0, 0, 1), 4.0f, 1e-5);

  // Activation is applied after accumulation.
  Tensor in4, out4;
  in4.allocate(Shape(1, 1, 1));
  in4.at(0, 0, 0) = -1.0f;
  out4.allocate(Shape(1, 1, 1));
  float one = 1.0f, zb = 0.0f;
  ConvParams p4 = {1, 1, 0, 1, Activation::Leaky};
  CHECK(be.conv2d(in4, &one, &zb, p4, out4) == Status::Ok);
  CHECK_NEAR(out4.at(0, 0, 0), -0.1f, 1e-6);

  // Wrong output shape is rejected and logged.
  Tensor wrong;
  wrong.allocate(Shape(2, 2, 1));
  {
    ScopedLog log(TinyYoloLogLevel::Error);
    CHECK(be.conv2d(in, w.data(), &bias, p, wrong) == Status::ShapeMismatch);
    CHECK(contains(log.out.text, "[E] Reference: conv2d: output"));
  }
}

void testPoolUpsampleConcat() {
  ReferenceBackend be;

  Tensor in, out;
  in.allocate(Shape(4, 4, 1));
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) in.at(y, x, 0) = (float)(y * 4 + x);
  out.allocate(Shape(2, 2, 1));
  CHECK(be.maxPool(in, 2, 2, 0, out) == Status::Ok);
  CHECK_NEAR(out.at(0, 0, 0), 5.0f, 1e-6);
  CHECK_NEAR(out.at(1, 1, 0), 15.0f, 1e-6);

  Tensor small, big;
  small.allocate(Shape(2, 2, 1));
  small.at(0, 0, 0) = 1;
  small.at(0, 1, 0) = 2;
  small.at(1, 0, 0) = 3;
  small.at(1, 1, 0) = 4;
  big.allocate(Shape(4, 4, 1));
  CHECK(be.upsample(small, 2, big) == Status::Ok);
  CHECK_NEAR(big.at(0, 1, 0), 1.0f, 1e-6);
  CHECK_NEAR(big.at(0, 2, 0), 2.0f, 1e-6);
  CHECK_NEAR(big.at(3, 3, 0), 4.0f, 1e-6);

  Tensor a, b, cat;
  a.allocate(Shape(1, 2, 1));
  a.at(0, 0, 0) = 1;
  a.at(0, 1, 0) = 2;
  b.allocate(Shape(1, 2, 2));
  b.at(0, 0, 0) = 3;
  b.at(0, 0, 1) = 4;
  b.at(0, 1, 0) = 5;
  b.at(0, 1, 1) = 6;
  cat.allocate(Shape(1, 2, 3));
  CHECK(be.concat(a, b, cat) == Status::Ok);
  CHECK_NEAR(cat.at(0, 0, 0), 1, 1e-6);
  CHECK_NEAR(cat.at(0, 0, 1), 3, 1e-6);
  CHECK_NEAR(cat.at(0, 0, 2), 4, 1e-6);
  CHECK_NEAR(cat.at(0, 1, 0), 2, 1e-6);
  CHECK_NEAR(cat.at(0, 1, 1), 5, 1e-6);
  CHECK_NEAR(cat.at(0, 1, 2), 6, 1e-6);

  Tensor copied;
  copied.allocate(a.shape());
  CHECK(be.copy(a, copied) == Status::Ok);
  CHECK_NEAR(copied.at(0, 1, 0), 2, 1e-6);
}

// ---------------------------------------------------------------- Image and preprocessing

void testImageView() {
  const uint8_t big[2] = {0xF8, 0x00};     // RGB565 red, high byte first
  const uint8_t little[2] = {0x00, 0xF8};  // same colour, low byte first
  const uint8_t green[2] = {0x07, 0xE0};   // RGB565 green, high byte first
  float rgb[3];

  ImageView(big, 1, 1, PixelFormat::RGB565Big).rgbAt(0, 0, rgb);
  CHECK_NEAR(rgb[0], 1.0f, 1e-6);
  CHECK_NEAR(rgb[1], 0.0f, 1e-6);
  ImageView(little, 1, 1, PixelFormat::RGB565Little).rgbAt(0, 0, rgb);
  CHECK_NEAR(rgb[0], 1.0f, 1e-6);
  ImageView(green, 1, 1, PixelFormat::RGB565Big).rgbAt(0, 0, rgb);
  CHECK_NEAR(rgb[1], 1.0f, 1e-6);

  const uint8_t bgr[3] = {0, 0, 255};  // BGR: blue=0, green=0, red=255
  ImageView(bgr, 1, 1, PixelFormat::BGR888).rgbAt(0, 0, rgb);
  CHECK_NEAR(rgb[0], 1.0f, 1e-6);

  const uint8_t gray[1] = {255};
  ImageView(gray, 1, 1, PixelFormat::Gray8).rgbAt(0, 0, rgb);
  CHECK_NEAR(rgb[2], 1.0f, 1e-6);

  // Invalid views are rejected and logged.
  {
    ScopedLog log(TinyYoloLogLevel::Error);
    CHECK(ImageView(nullptr, 2, 2, PixelFormat::RGB888).validate() == Status::InvalidArgument);
    CHECK(ImageView(gray, 0, 2, PixelFormat::Gray8).validate() == Status::InvalidArgument);
    CHECK(ImageView(gray, 2, 1, PixelFormat::RGB888, 2).validate() == Status::InvalidArgument);
    CHECK(contains(log.out.text, "[E] Image: image data is null"));
  }
}

void testLetterbox() {
  // 4x2 solid red image into an 8x8 network input: scale 2, 2 rows of padding on each side.
  std::vector<uint8_t> pixels(4 * 2 * 3);
  for (size_t i = 0; i < pixels.size(); i += 3) {
    pixels[i] = 255;
    pixels[i + 1] = 0;
    pixels[i + 2] = 0;
  }
  ImageView image(pixels.data(), 4, 2, PixelFormat::RGB888);

  Tensor input;
  input.allocate(Shape(8, 8, 3));
  LetterboxInfo info;
  Preprocessor pre;
  CHECK(pre.letterbox(image, input, info) == Status::Ok);
  CHECK_NEAR(info.scale, 2.0f, 1e-6);
  CHECK_NEAR(info.pad_x, 0.0f, 1e-6);
  CHECK_NEAR(info.pad_y, 2.0f, 1e-6);

  CHECK_NEAR(input.at(4, 4, 0), 1.0f, 1e-5);  // content: red
  CHECK_NEAR(input.at(4, 4, 1), 0.0f, 1e-5);
  CHECK_NEAR(input.at(0, 0, 0), Preprocessor::padValue(), 1e-6);  // padding: grey
  CHECK_NEAR(input.at(1, 7, 2), Preprocessor::padValue(), 1e-6);
  CHECK_NEAR(input.at(2, 3, 0), 1.0f, 1e-5);  // first content row

  float x = 4.0f, y = 6.0f;
  info.toSource(x, y);
  CHECK_NEAR(x, 2.0f, 1e-5);
  CHECK_NEAR(y, 2.0f, 1e-5);

  // Single-channel input receives luma.
  Tensor gray;
  gray.allocate(Shape(8, 8, 1));
  CHECK(pre.letterbox(image, gray, info) == Status::Ok);
  CHECK_NEAR(gray.at(4, 4, 0), 0.299f, 1e-4);

  // Invalid channel count is rejected.
  Tensor bad;
  bad.allocate(Shape(8, 8, 2));
  {
    ScopedLog log(TinyYoloLogLevel::Error);
    CHECK(pre.letterbox(image, bad, info) == Status::InvalidArgument);
    CHECK(contains(log.out.text, "[E] Preprocessor: network input"));
  }
}

// ---------------------------------------------------------------- Detections

void testDetections() {
  CHECK_NEAR(intersectionOverUnion({0, 0, 4, 4, 1, 0}, {0, 0, 4, 4, 1, 0}), 1.0f, 1e-6);
  CHECK_NEAR(intersectionOverUnion({0, 0, 4, 4, 1, 0}, {10, 10, 4, 4, 1, 0}), 0.0f, 1e-6);
  CHECK_NEAR(intersectionOverUnion({0, 0, 4, 4, 1, 0}, {1, 0, 4, 4, 1, 0}), 12.0f / 20.0f, 1e-6);

  Detections d;
  CHECK(d.reserve(0) == Status::InvalidArgument);
  CHECK(d.reserve(2) == Status::Ok);
  Detection item = {1, 2, 3, 4, 0.5f, 7};
  CHECK(d.push(item));
  CHECK(d.push(item));
  CHECK(!d.push(item));  // full: no reallocation, push reports it
  CHECK(d.size() == 2 && d.capacity() == 2);
  CHECK(d[0].class_id == 7);
  d.clear();
  CHECK(d.size() == 0);
}

// ---------------------------------------------------------------- Model and detector helpers

// Network: 8x8x3 input -> 1x1 conv to 6 channels (1 anchor, 1 class) -> YOLO head.
// Weights: channel 4 (objectness) reacts to the red input value; channel 5 (class) is
// constant high. Only pixels with strong red produce detections.
const float kAnchor[2] = {4.0f, 4.0f};
const char* const kLabels[] = {"bright"};
const LayerSpec kProbeLayers[] = {
    LayerSpec::conv(6, 1, 1, Activation::Linear),
    LayerSpec::yolo(1, kAnchor, 1),
};
const NetworkSpec kProbeNet = {8, 8, 3, kProbeLayers, 2, kLabels, 1};

std::vector<float> probeWeights() {
  // conv weights OIHW [6][3][1][1] = 18 values, then 6 biases.
  std::vector<float> w(18 + 6, 0.0f);
  w[4 * 3 + 0] = 20.0f;  // objectness reads the red channel
  w[18 + 4] = -10.0f;    // objectness bias
  w[18 + 5] = 10.0f;     // class bias
  return w;
}

void makeRedImage(std::vector<uint8_t>& pixels, const std::vector<std::pair<int, int>>& spots) {
  pixels.assign(8 * 8 * 3, 0);
  for (const auto& s : spots) {
    const size_t idx = ((size_t)s.second * 8 + s.first) * 3;
    pixels[idx] = 255;
  }
}

// ---------------------------------------------------------------- Model

void testModelBuildAndRun() {
  static const float anchors_coarse[] = {8, 8, 16, 16};
  static const float anchors_fine[] = {4, 4, 6, 6};
  static const LayerSpec layers[] = {
      LayerSpec::conv(8, 3, 2),                                  // 0: 4x4x8 (input 8x8)
      LayerSpec::conv(16, 3, 2),                                 // 1: 2x2x16
      LayerSpec::conv(16, 3, 1),                                 // 2: 2x2x16
      LayerSpec::conv(14, 1, 1, Activation::Linear).input(2),    // 3: 2x2x14 (2 x (5+2))
      LayerSpec::yolo(2, anchors_coarse, 2),                     // 4: coarse head
      LayerSpec::upsample(2).input(2),                           // 5: 4x4x16
      LayerSpec::route(5, 0),                                    // 6: 4x4x24
      LayerSpec::conv(14, 3, 1),                                 // 7: 4x4x14
      LayerSpec::yolo(2, anchors_fine, 2),                       // 8: fine head
  };
  const NetworkSpec spec = {8, 8, 3, layers, 9, nullptr, 0};

  Model model;
  ReferenceBackend be;
  CHECK(!model.isBuilt());
  CHECK(model.build(spec, be) == Status::Ok);
  CHECK(model.isBuilt());
  CHECK(model.headCount() == 2);
  CHECK(model.head(0).output().shape() == Shape(2, 2, 14));
  CHECK(model.head(1).output().shape() == Shape(4, 4, 14));
  CHECK(model.layer(6).output().shape() == Shape(4, 4, 24));

  // Parameters: conv0 8*3*9+8, conv1 16*8*9+16, conv2 16*16*9+16, conv3 14*16+14, conv7 14*24*9+14.
  const size_t expected = (216 + 8) + (1152 + 16) + (2304 + 16) + (224 + 14) + (3024 + 14);
  CHECK(model.paramCount() == expected);

  std::vector<float> weights(expected, 0.01f);
  ArrayWeightSource source(weights.data(), weights.size());
  CHECK(model.loadWeights(source) == Status::Ok);

  model.input().fill(0.5f);
  CHECK(model.run() == Status::Ok);
  for (int h = 0; h < model.headCount(); ++h) {
    const Tensor& out = model.head(h).output();
    bool finite = true;
    for (size_t i = 0; i < out.size(); ++i) finite = finite && std::isfinite(out.data()[i]);
    CHECK(finite);
  }
}

void testModelErrors() {
  static const float anchors[] = {8, 8};
  // Head expects 1 x (5+3) = 8 channels but the conv provides 6.
  static const LayerSpec bad_head[] = {
      LayerSpec::conv(6, 1, 1, Activation::Linear),
      LayerSpec::yolo(3, anchors, 1),
  };
  Model model;
  ReferenceBackend be;
  {
    ScopedLog log(TinyYoloLogLevel::Error);
    CHECK(model.build({8, 8, 3, bad_head, 2, nullptr, 0}, be) == Status::InvalidModel);
    CHECK(contains(log.out.text, "[E] Yolo: head expects 8 channels"));
    CHECK(contains(log.out.text, "[E] Model: layer 1 (Yolo) cannot be configured"));
  }

  // Forward reference to a later layer is rejected.
  static const LayerSpec forward_ref[] = {LayerSpec::conv(4, 1, 1).input(3),
                                          LayerSpec::conv(4, 1, 1)};
  {
    ScopedLog log(TinyYoloLogLevel::Error);
    CHECK(model.build({4, 4, 3, forward_ref, 2, nullptr, 0}, be) == Status::InvalidModel);
    CHECK(contains(log.out.text, "not an earlier layer"));
  }

  // Unsupported channel count.
  {
    ScopedLog log(TinyYoloLogLevel::Error);
    CHECK(model.build({4, 4, 4, forward_ref, 2, nullptr, 0}, be) == Status::InvalidModel);
    CHECK(contains(log.out.text, "unsupported input channel count 4"));
  }

  // Running before a successful build fails.
  Model unbuilt;
  {
    ScopedLog log(TinyYoloLogLevel::Error);
    CHECK(unbuilt.run() == Status::NotInitialized);
    CHECK(contains(log.out.text, "run called before"));
  }

  // Too few weights.
  CHECK(model.build(kProbeNet, be) == Status::Ok);
  std::vector<float> weights = probeWeights();
  weights.pop_back();
  ArrayWeightSource short_source(weights.data(), weights.size());
  {
    ScopedLog log(TinyYoloLogLevel::Error);
    CHECK(model.loadWeights(short_source) == Status::WeightError);
    CHECK(contains(log.out.text, "[E] Weights: array exhausted"));
    CHECK(contains(log.out.text, "[E] Model: layer 0 (Conv): reading weights failed"));
  }
}

void testWeightSources() {
  const float values[3] = {1.5f, -2.0f, 3.25f};
  ArrayWeightSource array(values, 3);
  float dst[2];
  CHECK(array.read(dst, 2) == Status::Ok);
  CHECK_NEAR(dst[1], -2.0f, 1e-6);
  CHECK(array.remaining() == 1);

  MemoryStream stream(values, sizeof(values));
  StreamWeightSource from_stream(stream);
  float out[3];
  CHECK(from_stream.read(out, 3) == Status::Ok);
  CHECK_NEAR(out[2], 3.25f, 1e-6);

  MemoryStream truncated(values, 4);
  StreamWeightSource short_stream(truncated);
  {
    ScopedLog log(TinyYoloLogLevel::Error);
    CHECK(short_stream.read(out, 3) == Status::WeightError);
    CHECK(contains(log.out.text, "stream ended after 4 of 12 bytes"));
  }
}

// ---------------------------------------------------------------- Detector end to end

void testDetectorDecodesBoxes() {
  Detector detector;
  std::vector<float> weights = probeWeights();
  ArrayWeightSource source(weights.data(), weights.size());
  DetectorConfig config;
  CHECK(detector.begin(kProbeNet, source, config) == Status::Ok);
  CHECK(detector.isReady());
  CHECK(strcmp(detector.label(0), "bright") == 0);

  std::vector<uint8_t> pixels;
  Detections out;
  CHECK(out.reserve(10) == Status::Ok);

  // One bright pixel at column 3, row 5: one box centred on that cell.
  makeRedImage(pixels, {{3, 5}});
  ImageView image(pixels.data(), 8, 8, PixelFormat::RGB888);
  CHECK(detector.detect(image, out) == Status::Ok);
  CHECK(out.size() == 1);
  if (out.size() == 1) {
    const Detection& d = out[0];
    CHECK_NEAR(d.x, 1.5f, 1e-4);
    CHECK_NEAR(d.y, 3.5f, 1e-4);
    CHECK_NEAR(d.width, 4.0f, 1e-4);
    CHECK_NEAR(d.height, 4.0f, 1e-4);
    CHECK(d.score > 0.99f);
    CHECK(d.class_id == 0);
  }

  // Two neighbouring spots overlap with IoU 0.6: NMS at 0.45 keeps one, at 0.7 keeps two.
  makeRedImage(pixels, {{3, 5}, {4, 5}});
  CHECK(detector.detect(image, out) == Status::Ok);
  CHECK(out.size() == 1);

  DetectorConfig loose;
  loose.iou_threshold = 0.7f;
  std::vector<float> weights2 = probeWeights();
  ArrayWeightSource source2(weights2.data(), weights2.size());
  CHECK(detector.begin(kProbeNet, source2, loose) == Status::Ok);
  CHECK(detector.detect(image, out) == Status::Ok);
  CHECK(out.size() == 2);

  // Empty image: nothing passes the threshold.
  makeRedImage(pixels, {});
  CHECK(detector.detect(image, out) == Status::Ok);
  CHECK(out.size() == 0);
}

void testDetectorBoundaryClipping() {
  // A spot in the top-left corner: its 4x4 box is clipped to the image.
  std::vector<float> weights = probeWeights();
  ArrayWeightSource source(weights.data(), weights.size());
  Detector detector;
  CHECK(detector.begin(kProbeNet, source) == Status::Ok);

  std::vector<uint8_t> pixels;
  makeRedImage(pixels, {{0, 0}});
  ImageView image(pixels.data(), 8, 8, PixelFormat::RGB888);
  Detections out;
  out.reserve(4);
  CHECK(detector.detect(image, out) == Status::Ok);
  CHECK(out.size() == 1);
  if (out.size() == 1) {
    CHECK_NEAR(out[0].x, 0.0f, 1e-4);
    CHECK_NEAR(out[0].y, 0.0f, 1e-4);
    CHECK(out[0].width > 0.0f && out[0].width <= 8.0f);
    CHECK(out[0].x + out[0].width <= 8.0f + 1e-4f);
  }
}

void testDetectorWarningsAndErrors() {
  std::vector<float> weights = probeWeights();
  ArrayWeightSource source(weights.data(), weights.size());
  Detector detector;

  {
    ScopedLog log(TinyYoloLogLevel::Warning);
    DetectorConfig tiny;
    tiny.max_candidates = 1;
    CHECK(detector.begin(kProbeNet, source, tiny) == Status::Ok);
    std::vector<uint8_t> pixels;
    makeRedImage(pixels, {{3, 5}, {4, 5}});  // two candidates, room for one
    ImageView image(pixels.data(), 8, 8, PixelFormat::RGB888);
    Detections out;
    out.reserve(4);
    CHECK(detector.detect(image, out) == Status::Ok);
    CHECK(contains(log.out.text, "[W] Postprocessor: 1 candidate boxes dropped"));
  }

  {
    ScopedLog log(TinyYoloLogLevel::Warning);
    DetectorConfig bad;
    bad.iou_threshold = 3.0f;
    bad.score_threshold = 0.0f;
    std::vector<float> fresh_weights = probeWeights();  // begin() consumes its source
    ArrayWeightSource fresh_source(fresh_weights.data(), fresh_weights.size());
    CHECK(detector.begin(kProbeNet, fresh_source, bad) == Status::Ok);
    CHECK(detector.config().iou_threshold == 1.0f);
    CHECK(detector.config().score_threshold > 0.0f);
    CHECK(contains(log.out.text, "[W] Detector: iou_threshold"));
    CHECK(contains(log.out.text, "[W] Detector: score_threshold"));
  }

  {
    ScopedLog log(TinyYoloLogLevel::Warning);
    CHECK(detector.label(5) != nullptr);
    CHECK(strcmp(detector.label(5), "?") == 0);
    CHECK(contains(log.out.text, "[W] Detector: no label for class 5"));
  }

  {
    ScopedLog log(TinyYoloLogLevel::Error);
    Detector fresh;
    Detections out;
    out.reserve(2);
    std::vector<uint8_t> pixels(8 * 8 * 3, 0);
    ImageView image(pixels.data(), 8, 8, PixelFormat::RGB888);
    CHECK(fresh.detect(image, out) == Status::NotInitialized);
    CHECK(contains(log.out.text, "[E] Detector: detect called before"));
  }

  {
    ScopedLog log(TinyYoloLogLevel::Error);
    Detections empty;
    std::vector<uint8_t> pixels(8 * 8 * 3, 0);
    ImageView image(pixels.data(), 8, 8, PixelFormat::RGB888);
    CHECK(detector.detect(image, empty) == Status::InvalidArgument);
    CHECK(contains(log.out.text, "[E] Detector: output container has no capacity"));
  }

  {
    ScopedLog log(TinyYoloLogLevel::Error);
    DetectorConfig zero;
    zero.max_candidates = 0;
    std::vector<float> zero_weights = probeWeights();
    ArrayWeightSource zero_source(zero_weights.data(), zero_weights.size());
    CHECK(detector.begin(kProbeNet, zero_source, zero) == Status::InvalidArgument);
    CHECK(!detector.isReady());
    CHECK(contains(log.out.text, "[E] Detector: max_candidates must be positive"));
  }
}

void testLabelCountWarning() {
  std::vector<float> weights = probeWeights();
  ArrayWeightSource source(weights.data(), weights.size());
  static const char* const two_labels[] = {"a", "b"};
  const NetworkSpec spec = {8, 8, 3, kProbeLayers, 2, two_labels, 2};
  Detector detector;
  ScopedLog log(TinyYoloLogLevel::Warning);
  CHECK(detector.begin(spec, source) == Status::Ok);
  CHECK(contains(log.out.text, "[W] Detector: 2 labels given but the head predicts 1 classes"));
}

void testFloatReadOverRealisticSize() {
  // A 64x64 network similar to the example sketch, end to end on synthetic data.
  static const float anchors[] = {8, 8, 16, 16, 32, 32};
  static const LayerSpec layers[] = {
      LayerSpec::conv(16, 3, 2), LayerSpec::conv(32, 3, 2), LayerSpec::conv(64, 3, 2),
      LayerSpec::conv(24, 1, 1, Activation::Linear), LayerSpec::yolo(3, anchors, 3)};
  static const char* const labels[] = {"person", "car", "dog"};
  const NetworkSpec spec = {64, 64, 3, layers, 5, labels, 3};

  Detector detector;
  Model probe;
  ReferenceBackend be;
  CHECK(probe.build(spec, be) == Status::Ok);
  std::vector<float> weights(probe.paramCount());
  unsigned int seed = 12345;
  for (size_t i = 0; i < weights.size(); ++i) {
    seed = seed * 1103515245u + 12345u;
    weights[i] = (float)((seed >> 16) & 0xFFFF) / 65535.0f * 0.1f - 0.05f;
  }
  ArrayWeightSource source(weights.data(), weights.size());
  CHECK(detector.begin(spec, source) == Status::Ok);

  std::vector<uint8_t> pixels(64 * 64 * 3);
  for (int y = 0; y < 64; ++y)
    for (int x = 0; x < 64; ++x) {
      uint8_t* p = &pixels[((size_t)y * 64 + x) * 3];
      p[0] = (uint8_t)(x * 4);
      p[1] = (uint8_t)(y * 4);
      p[2] = 128;
    }
  ImageView image(pixels.data(), 64, 64, PixelFormat::RGB888);
  Detections out;
  out.reserve(20);
  CHECK(detector.detect(image, out) == Status::Ok);
  for (const Detection& d : out) {
    CHECK(d.x >= 0.0f && d.y >= 0.0f);
    CHECK(d.x + d.width <= 64.0f + 1e-3f && d.y + d.height <= 64.0f + 1e-3f);
    CHECK(d.score >= detector.config().score_threshold);
    CHECK(d.class_id >= 0 && d.class_id < 3);
  }
}

// ---------------------------------------------------------------- Darknet models, flash and SD sources

// Small YOLOv3-tiny-shaped network: two heads, a route with one input, an upsample and a
// concatenation with an earlier layer.
const char kMiniCfg[] = R"cfg(
[net]
width=16
height=16
channels=3

[convolutional]
batch_normalize=1
filters=8
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=16
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=16
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=14
activation=linear

[yolo]
mask=0,1
anchors=4,4, 6,6, 8,8, 12,12
classes=2

[route]
layers=-3

[upsample]
stride=2

[route]
layers=-1,0

[convolutional]
size=3
stride=1
pad=1
filters=14
activation=linear

[yolo]
mask=2,3
anchors=4,4, 6,6, 8,8, 12,12
classes=2
)cfg";

void appendBytes(std::vector<uint8_t>& out, const void* data, size_t n) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  out.insert(out.end(), p, p + n);
}

void testMaxPoolDarknetStyle() {
  // size 2, stride 1, total padding 1: same resolution, window starts at the pixel itself.
  ReferenceBackend be;
  Tensor in, out;
  in.allocate(Shape(4, 4, 1));
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) in.at(y, x, 0) = (float)(y * 4 + x);
  out.allocate(Shape(4, 4, 1));
  CHECK(be.maxPool(in, 2, 1, 1, out) == Status::Ok);
  CHECK_NEAR(out.at(0, 0, 0), 5.0f, 1e-6);
  CHECK_NEAR(out.at(3, 3, 0), 15.0f, 1e-6);
  CHECK(poolOutputSize(13, 2, 1, 1) == 13);
  CHECK(poolOutputSize(416, 2, 2, 1) == 208);
}

void testDarknetDecodeStyle() {
  // Darknet decoding with exp(tw) = 2 and anchor 2 gives a 4-pixel box, not the 1.78-pixel
  // box YOLOv5-style decoding would give for the same raw value.
  static const float anchor[2] = {2.0f, 2.0f};
  static const LayerSpec layers[] = {
      LayerSpec::conv(6, 1, 1, Activation::Linear),
      LayerSpec::yolo(1, anchor, 1, YoloDecode::Darknet),
  };
  const NetworkSpec spec = {8, 8, 3, layers, 2, nullptr, 0};
  std::vector<float> weights = probeWeights();
  weights[18 + 2] = weights[18 + 3] = std::log(2.0f);  // tw, th bias
  ArrayWeightSource source(weights.data(), weights.size());
  Detector detector;
  CHECK(detector.begin(spec, source) == Status::Ok);

  std::vector<uint8_t> pixels;
  makeRedImage(pixels, {{3, 5}});
  ImageView image(pixels.data(), 8, 8, PixelFormat::RGB888);
  Detections out;
  out.reserve(4);
  CHECK(detector.detect(image, out) == Status::Ok);
  CHECK(out.size() == 1);
  if (out.size() == 1) {
    CHECK_NEAR(out[0].x, 1.5f, 1e-4);
    CHECK_NEAR(out[0].y, 3.5f, 1e-4);
    CHECK_NEAR(out[0].width, 4.0f, 1e-4);
    CHECK_NEAR(out[0].height, 4.0f, 1e-4);
  }
}

void testCfgBuildsAndRuns() {
  BufferStream cfg(kMiniCfg, sizeof(kMiniCfg) - 1);  // flash-style source
  NetworkDescription desc;
  CHECK(desc.parseCfg(cfg) == Status::Ok);
  CHECK(desc.isValid());
  const NetworkSpec& spec = desc.spec();
  CHECK(spec.input_w == 16 && spec.input_h == 16 && spec.input_c == 3);
  CHECK(spec.layer_count == 10);
  if (spec.layer_count != 10) return;
  CHECK(spec.layers[5].type == LayerType::Route && spec.layers[5].from[0] == 2);
  CHECK(spec.layers[5].from[1] == kNone);
  CHECK(spec.layers[7].from[0] == 6 && spec.layers[7].from[1] == 0);
  CHECK(spec.layers[4].decode == YoloDecode::Darknet);
  CHECK(spec.layers[4].num_anchors == 2 && spec.layers[4].anchors[0] == 4.0f);
  CHECK(spec.layers[9].anchors[0] == 8.0f && spec.layers[9].anchors[2] == 12.0f);
  CHECK(spec.layers[0].batch_norm && !spec.layers[8].batch_norm);

  Model model;
  ReferenceBackend be;
  CHECK(model.build(spec, be) == Status::Ok);
  CHECK(model.headCount() == 2);
  CHECK(model.head(0).output().shape() == Shape(4, 4, 14));
  CHECK(model.head(1).output().shape() == Shape(8, 8, 14));
  CHECK(model.layer(6).output().shape() == Shape(8, 8, 16));
  CHECK(model.layer(7).output().shape() == Shape(8, 8, 24));

  std::vector<float> weights(model.paramCount(), 0.01f);
  ArrayWeightSource source(weights.data(), weights.size());
  CHECK(model.loadWeights(source) == Status::Ok);
  model.input().fill(0.5f);
  CHECK(model.run() == Status::Ok);
}

void testDarknetBatchNormFolding() {
  static const char kCfg[] =
      "[net]\nwidth=8\nheight=8\nchannels=3\n"
      "[convolutional]\nbatch_normalize=1\nfilters=6\nsize=1\nstride=1\npad=0\nactivation=linear\n"
      "[yolo]\nmask=0\nanchors=4,4\nclasses=1\n";
  BufferStream cfg(kCfg, sizeof(kCfg) - 1);
  NetworkDescription desc;
  CHECK(desc.parseCfg(cfg) == Status::Ok);
  Model model;
  ReferenceBackend be;
  CHECK(model.build(desc.spec(), be) == Status::Ok);

  // Darknet .weights: header (0.2), 64-bit "seen", then beta, gamma, mean, variance, weights.
  std::vector<uint8_t> file;
  const uint32_t version[3] = {0, 2, 0};
  const uint64_t seen = 0;
  appendBytes(file, version, sizeof(version));
  appendBytes(file, &seen, sizeof(seen));
  std::vector<float> body;
  for (int i = 0; i < 6; ++i) body.push_back(1.0f);     // beta
  for (int i = 0; i < 6; ++i) body.push_back(2.0f);     // gamma
  for (int i = 0; i < 6; ++i) body.push_back(0.25f);    // running mean
  for (int i = 0; i < 6; ++i) body.push_back(0.99999f); // running variance (sqrt(var+eps) = 1)
  for (int i = 0; i < 18; ++i) body.push_back(i < 3 ? 1.0f : 0.0f);  // output 0 only
  appendBytes(file, body.data(), body.size() * sizeof(float));

  BufferStream weights_stream(file.data(), file.size());
  DarknetWeightSource source(weights_stream);
  CHECK(source.open() == Status::Ok);
  CHECK(model.loadWeights(source) == Status::Ok);
  CHECK(source.finish() == Status::Ok);

  model.input().fill(1.0f);
  CHECK(model.run() == Status::Ok);
  // Output 0: 3 inputs x (1 * s) + bias, with s = gamma/sqrt(var+eps) = 2 and bias = beta - s*mean.
  CHECK_NEAR(model.layer(0).output().at(0, 0, 0), 3 * 2.0f + 0.5f, 1e-4);
  CHECK_NEAR(model.layer(0).output().at(0, 0, 1), 0.5f, 1e-4);

  // Extra bytes at the end are reported as a warning.
  std::vector<uint8_t> longer = file;
  longer.push_back(0);
  BufferStream long_stream(longer.data(), longer.size());
  DarknetWeightSource long_source(long_stream);
  ScopedLog log(TinyYoloLogLevel::Warning);
  CHECK(long_source.open() == Status::Ok);
  Model again;
  CHECK(again.build(desc.spec(), be) == Status::Ok);
  CHECK(again.loadWeights(long_source) == Status::Ok);
  CHECK(long_source.finish() == Status::Ok);
  CHECK(contains(log.out.text, "[W] Darknet: weight file has unread data"));
}

void testDarknetWeightsTruncated() {
  static const char kCfg[] =
      "[net]\nwidth=8\nheight=8\nchannels=3\n"
      "[convolutional]\nbatch_normalize=1\nfilters=6\nsize=1\nstride=1\nactivation=linear\n"
      "[yolo]\nmask=0\nanchors=4,4\nclasses=1\n";
  BufferStream cfg(kCfg, sizeof(kCfg) - 1);
  NetworkDescription desc;
  CHECK(desc.parseCfg(cfg) == Status::Ok);
  Model model;
  ReferenceBackend be;
  CHECK(model.build(desc.spec(), be) == Status::Ok);

  const uint8_t header[12 + 8] = {0};  // version 0.0 is rejected later by the weight count only
  BufferStream stream(header, sizeof(header));
  DarknetWeightSource source(stream);
  ScopedLog log(TinyYoloLogLevel::Error);
  CHECK(source.open() == Status::Ok);
  CHECK(model.loadWeights(source) == Status::WeightError);
  CHECK(contains(log.out.text, "[E] Darknet: weight stream ended"));
  CHECK(contains(log.out.text, "[E] Model: layer 0 (Conv): reading weights failed"));
}

void testFlashNamesAndBufferStream() {
  static const uint8_t kNamesBytes[] = "person\ncar\n\ndog\r\n";
  BufferStream names(kNamesBytes, sizeof(kNamesBytes) - 1);
  BufferStream cfg(kMiniCfg, sizeof(kMiniCfg) - 1);
  NetworkDescription desc;
  CHECK(desc.parseCfg(cfg) == Status::Ok);
  CHECK(desc.parseNames(names) == Status::Ok);
  const NetworkSpec& spec = desc.spec();
  CHECK(spec.label_count == 3);
  if (spec.label_count == 3) {
    CHECK(strcmp(spec.labels[0], "person") == 0);
    CHECK(strcmp(spec.labels[1], "car") == 0);
    CHECK(strcmp(spec.labels[2], "dog") == 0);
  }

  uint8_t out[4];
  BufferStream raw(kNamesBytes, 4);
  CHECK(raw.readBytes((char*)out, 4) == 4);
  CHECK(out[0] == 'p' && raw.remaining() == 0 && raw.read() == -1);
  raw.rewind();
  CHECK(raw.peek() == 'p' && raw.available() == 4);
}

void testDarknetCfgErrors() {
  struct Case {
    const char* text;
    Status expected;
    const char* log_fragment;
  };
  static const Case cases[] = {
      {"[net]\nwidth=8\nheight=8\nchannels=3\n", Status::InvalidModel, "defines no layers"},
      {"[convolutional]\nfilters=1\n", Status::InvalidModel, "appears before [net]"},
      {"[net]\nwidth=eight\n", Status::InvalidModel, "'width' must be an integer"},
      {"width=8\n", Status::InvalidModel, "outside of a [section]"},
      {"[net]\nwidth=8\nheight=8\nchannels=3\n[reorg]\nfrom=-3\n", Status::Unsupported,
       "section [reorg] is not supported"},
      {"[net]\nwidth=8\nheight=8\nchannels=3\n[convolutional]\nfilters=4\nsize=3\nstride=1\n"
       "activation=swish\n",
       Status::Unsupported, "activation 'swish' is not supported"},
      {"[net]\nwidth=8\nheight=8\nchannels=3\n[convolutional]\nfilters=4\nsize=1\nstride=1\n"
       "activation=linear\n[route]\nlayers=-1,-1,-1\n",
       Status::Unsupported, "route with 3 inputs is not supported"},
  };
  for (const Case& c : cases) {
    ScopedLog log(TinyYoloLogLevel::Error);
    BufferStream stream(c.text, strlen(c.text));
    NetworkDescription desc;
    CHECK(desc.parseCfg(stream) == c.expected);
    CHECK(!desc.isValid());
    if (!contains(log.out.text, c.log_fragment)) {
      fprintf(stderr, "expected log '%s', got:\n%s", c.log_fragment, log.out.text.c_str());
    }
    CHECK(contains(log.out.text, c.log_fragment));
  }
}

// ---------------------------------------------------------------- Optimised backends

float unitRandom(unsigned& seed) {
  seed = seed * 1664525u + 1013904223u;
  return ((seed >> 8) & 0xFFFF) / 32767.5f - 1.0f;  // [-1, 1]
}

/// Compares `fast` against ReferenceBackend for one convolution shape.
void checkConvMatches(Backend& fast, int H, int W, int C, int OC, int K, int S, int P,
                      Activation act) {
  unsigned seed = (unsigned)(H * 131 + W * 17 + C * 7 + OC * 3 + K);
  Tensor in;
  CHECK(in.allocate(Shape(H, W, C)) == Status::Ok);
  for (size_t i = 0; i < in.size(); ++i) in.data()[i] = unitRandom(seed);
  std::vector<float> weights((size_t)OC * C * K * K);
  for (float& w : weights) w = unitRandom(seed) * 0.5f;
  std::vector<float> bias(OC);
  for (float& b : bias) b = unitRandom(seed);

  const ConvParams p = {K, S, P, OC, act};
  const int oh = windowOutputSize(H, K, S, P);
  const int ow = windowOutputSize(W, K, S, P);
  Tensor expected, actual;
  CHECK(expected.allocate(Shape(oh, ow, OC)) == Status::Ok);
  CHECK(actual.allocate(Shape(oh, ow, OC)) == Status::Ok);
  ReferenceBackend reference;
  CHECK(reference.conv2d(in, weights.data(), bias.data(), p, expected) == Status::Ok);
  CHECK(fast.conv2d(in, weights.data(), bias.data(), p, actual) == Status::Ok);
  double worst = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double diff = fabs((double)expected.data()[i] - (double)actual.data()[i]);
    if (diff > worst) worst = diff;
  }
  CHECK(worst < 1e-4);
}

void testConvolutionAgreesWithReference() {
  struct Shape4 {
    int H, W, C, OC, K, S, P;
    Activation act;
  };
  static const Shape4 shapes[] = {
      {13, 11, 5, 7, 3, 2, 1, Activation::Leaky},   // strided, padded, odd sizes
      {9, 9, 3, 4, 1, 1, 0, Activation::Linear},    // 1x1
      {8, 8, 2, 3, 5, 1, 2, Activation::ReLU},      // large kernel
      {16, 16, 4, 6, 3, 1, 1, Activation::Leaky},   // same-size
      {1, 1, 1, 1, 1, 1, 0, Activation::Linear},    // single pixel, fewer rows than workers
      {20, 6, 8, 16, 3, 2, 0, Activation::SiLU},    // no padding
  };
  for (const Shape4& s : shapes) {
    for (int workers : {1, 2, 3, 4}) {
      DotProductBackend dot(workers);
      checkConvMatches(dot, s.H, s.W, s.C, s.OC, s.K, s.S, s.P, s.act);
    }
    Esp32Backend esp;  // falls back to the scalar dot product off the ESP32
    checkConvMatches(esp, s.H, s.W, s.C, s.OC, s.K, s.S, s.P, s.act);
  }
}

void coverRange(void* ctx, int begin, int end, int) {
  int* hits = static_cast<int*>(ctx);
  for (int i = begin; i < end; ++i) hits[i] += 1;
}

void testParallelRangeCoversEachIndexOnce() {
  for (int workers : {1, 2, 3, 4, 8}) {
    for (int count : {1, 2, 7, 10, 64}) {
      std::vector<int> hits((size_t)count, 0);
      parallelRange(count, workers, coverRange, hits.data());
      bool once = true;
      for (int h : hits) once = once && h == 1;
      CHECK(once);
    }
  }
  CHECK(hardwareWorkers() >= 1 && hardwareWorkers() <= kMaxWorkers);
}

void testLargeBufferAllocation() {
  // On the desktop this is plain calloc; on ESP32 it would be PSRAM when available.
  Buffer<float> big;
  CHECK(big.allocate(40 * 1024) == Status::Ok);
  CHECK(big.size() == 40 * 1024);
  bool all_zero = true;
  for (size_t i = 0; i < big.size(); ++i) all_zero = all_zero && big[i] == 0.0f;
  CHECK(all_zero);
  Buffer<float> moved = std::move(big);
  CHECK(moved.size() == 40 * 1024 && big.size() == 0);
}

void testDotProductNetworkMatchesReference() {
  // The Darknet mini network, run once with the reference backend and once with the
  // dot-product backend on three threads. Every head output must agree.
  NetworkDescription desc;
  BufferStream cfg(kMiniCfg, sizeof(kMiniCfg) - 1);
  CHECK(desc.parseCfg(cfg) == Status::Ok);

  ReferenceBackend reference;
  DotProductBackend threaded(3);
  Model a, b;
  CHECK(a.build(desc.spec(), reference) == Status::Ok);
  CHECK(b.build(desc.spec(), threaded) == Status::Ok);

  std::vector<float> weights(a.paramCount());
  unsigned seed = 7;
  for (float& w : weights) w = unitRandom(seed) * 0.3f;
  ArrayWeightSource sa(weights.data(), weights.size());
  ArrayWeightSource sb(weights.data(), weights.size());
  CHECK(a.loadWeights(sa) == Status::Ok);
  CHECK(b.loadWeights(sb) == Status::Ok);

  for (size_t i = 0; i < a.input().size(); ++i) {
    const float v = 0.5f + 0.5f * unitRandom(seed);
    a.input().data()[i] = v;
    b.input().data()[i] = v;
  }
  CHECK(a.run() == Status::Ok);
  CHECK(b.run() == Status::Ok);
  CHECK(a.headCount() == b.headCount());
  for (int h = 0; h < a.headCount(); ++h) {
    const Tensor& ta = a.head(h).output();
    const Tensor& tb = b.head(h).output();
    CHECK(ta.shape() == tb.shape());
    double worst = 0.0;
    for (size_t i = 0; i < ta.size(); ++i) {
      const double diff = fabs((double)ta.data()[i] - (double)tb.data()[i]);
      if (diff > worst) worst = diff;
    }
    CHECK(worst < 1e-4);
  }
}

void testDrawRectangle() {
  // 10x10 RGB888 image, all black. Box [1, 8] x [1, 8], red, 1 px and 2 px thick.
  std::vector<uint8_t> img(10 * 10 * 3, 0);
  const Rgb red = {255, 0, 0};
  auto px = [&img](int x, int y, int c) { return img[((size_t)y * 10 + x) * 3 + c]; };

  drawRectangle(img.data(), 10, 10, 1, 1, 8, 8, red, 1);
  CHECK(px(1, 1, 0) == 255 && px(8, 8, 0) == 255 && px(1, 5, 0) == 255 && px(8, 5, 0) == 255);
  CHECK(px(4, 1, 0) == 255 && px(4, 8, 0) == 255);
  CHECK(px(4, 4, 0) == 0 && px(4, 2, 0) == 0);  // interior untouched
  CHECK(px(0, 0, 0) == 0 && px(9, 9, 0) == 0);  // outside untouched

  std::fill(img.begin(), img.end(), 0);
  drawRectangle(img.data(), 10, 10, 1, 1, 8, 8, red, 2);
  CHECK(px(4, 2, 0) == 255 && px(2, 4, 0) == 255);  // second line inwards
  CHECK(px(4, 3, 0) == 0);

  // Boxes partly outside the image are clipped, not crashing.
  std::fill(img.begin(), img.end(), 0);
  drawRectangle(img.data(), 10, 10, -5, -5, 3, 3, red, 1);
  CHECK(px(3, 0, 0) == 255 && px(0, 3, 0) == 255);

  // Detection in source pixels: x=2, y=2, 4x4 -> corners at 2 and 5.
  std::fill(img.begin(), img.end(), 0);
  Detection d = {2.0f, 2.0f, 4.0f, 4.0f, 0.9f, 0};
  drawDetection(img.data(), 10, 10, d, red, 1);
  CHECK(px(2, 2, 0) == 255 && px(5, 5, 0) == 255 && px(5, 2, 0) == 255);
  CHECK(px(6, 6, 0) == 0);
}

void testTaskRunsAndEnds() {
  std::atomic<int> value(0);
  Task task;
  CHECK(!task.isRunning() && !task.isEnded());
  CHECK(task.begin([&value]() { value.store(42); }));
  task.end();
  CHECK(value.load() == 42);
  CHECK(task.isEnded() && !task.isRunning());

  // A second run after the first has ended.
  CHECK(task.begin([&value]() { value.store(7); }));
  task.end();
  CHECK(value.load() == 7);

  // Destruction waits for a running function: no use-after-free of the captured flag.
  std::atomic<bool> finished(false);
  {
    Task scoped;
    CHECK(scoped.begin([&finished]() { finished.store(true); }));
  }
  CHECK(finished.load());
}

void testQuantizeRows() {
  const float w[8] = {0.5f, -1.0f, 0.25f, 0.0f,   // row 0: max 1.0
                      0.0f, 0.0f, 0.0f, 0.0f};    // row 1: all zero
  int8_t q[8];
  float scales[2];
  quantizeRows(w, 2, 4, q, scales);
  CHECK_NEAR(scales[0], 1.0f / 127.0f, 1e-7);
  CHECK(scales[1] == 1.0f);
  CHECK(q[1] == -127);                       // -1.0 maps to -127
  CHECK(q[0] == 64);                         // 0.5 / (1/127) = 63.5 -> 64
  for (int i = 0; i < 4; ++i) {
    CHECK(fabs(w[i] - q[i] * scales[0]) <= scales[0] * 0.5f + 1e-7f);
  }
  for (int i = 4; i < 8; ++i) CHECK(q[i] == 0);
}

/// Reorders int8 conv weights from OIHW ([o][i][ky][kx]) to OHWI ([o][ky][kx][i]).
std::vector<int8_t> toOhwi(const std::vector<int8_t>& oihw, int OC, int IC, int K) {
  std::vector<int8_t> ohwi(oihw.size());
  for (int o = 0; o < OC; ++o)
    for (int i = 0; i < IC; ++i)
      for (int ky = 0; ky < K; ++ky)
        for (int kx = 0; kx < K; ++kx)
          ohwi[((o * K + ky) * K + kx) * IC + i] = oihw[((o * IC + i) * K + ky) * K + kx];
  return ohwi;
}

void testConvInt8MatchesDequantizedFloat() {
  // Int8 convolution must equal float convolution with the dequantized weights.
  unsigned seed = 99;
  const int H = 9, W = 7, C = 4, OC = 5, K = 3, S = 2, P = 1;
  Tensor in, out_int8, out_float;
  CHECK(in.allocate(Shape(H, W, C)) == Status::Ok);
  for (size_t i = 0; i < in.size(); ++i) in.data()[i] = unitRandom(seed);
  std::vector<float> w((size_t)OC * C * K * K);
  for (float& v : w) v = unitRandom(seed);
  std::vector<float> bias(OC);
  for (float& b : bias) b = unitRandom(seed) * 0.1f;

  std::vector<int8_t> q(w.size());
  std::vector<float> scales(OC);
  quantizeRows(w.data(), OC, w.size() / OC, q.data(), scales.data());
  std::vector<float> dequant(w.size());
  for (size_t i = 0; i < w.size(); ++i) dequant[i] = q[i] * scales[i / (w.size() / OC)];
  const std::vector<int8_t> q_ohwi = toOhwi(q, OC, C, K);  // the int8 kernels take OHWI

  const int oh = windowOutputSize(H, K, S, P), ow = windowOutputSize(W, K, S, P);
  CHECK(out_int8.allocate(Shape(oh, ow, OC)) == Status::Ok);
  CHECK(out_float.allocate(Shape(oh, ow, OC)) == Status::Ok);
  const ConvParams p = {K, S, P, OC, Activation::Leaky};
  ReferenceBackend reference;
  DotProductBackend threaded(3);
  Esp32Backend esp;
  CHECK(reference.conv2d(in, dequant.data(), bias.data(), p, out_float) == Status::Ok);
  for (Backend* be : {static_cast<Backend*>(&reference), static_cast<Backend*>(&threaded),
                      static_cast<Backend*>(&esp)}) {
    CHECK(be->conv2dInt8(in, q_ohwi.data(), scales.data(), bias.data(), p, out_int8) == Status::Ok);
    double worst = 0.0;
    for (size_t i = 0; i < out_float.size(); ++i) {
      worst = std::max(worst, fabs((double)out_float.data()[i] - (double)out_int8.data()[i]));
    }
    CHECK(worst < 1e-4);
  }
}

void testModelInt8CloseToFloat() {
  NetworkDescription desc;
  BufferStream cfg(kMiniCfg, sizeof(kMiniCfg) - 1);
  CHECK(desc.parseCfg(cfg) == Status::Ok);
  ReferenceBackend be;
  Model float_model, int8_model;
  CHECK(float_model.build(desc.spec(), be, WeightPrecision::Float32) == Status::Ok);
  CHECK(int8_model.build(desc.spec(), be, WeightPrecision::Int8) == Status::Ok);
  CHECK(int8_model.precision() == WeightPrecision::Int8);

  // Smaller, but not by a quarter: the layers that feed a YOLO head stay float32.
  CHECK(int8_model.weightBytes() < float_model.weightBytes());

  std::vector<float> weights(float_model.paramCount());
  unsigned seed = 5;
  for (float& w : weights) w = unitRandom(seed) * 0.3f;
  ArrayWeightSource s1(weights.data(), weights.size());
  ArrayWeightSource s2(weights.data(), weights.size());
  CHECK(float_model.loadWeights(s1) == Status::Ok);
  CHECK(int8_model.loadWeights(s2) == Status::Ok);
  for (size_t i = 0; i < float_model.input().size(); ++i) {
    const float v = 0.5f + 0.5f * unitRandom(seed);
    float_model.input().data()[i] = v;
    int8_model.input().data()[i] = v;
  }
  CHECK(float_model.run() == Status::Ok);
  CHECK(int8_model.run() == Status::Ok);
  for (int h = 0; h < float_model.headCount(); ++h) {
    const Tensor& a = float_model.head(h).output();
    const Tensor& b = int8_model.head(h).output();
    double range = 0.0, worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      range = std::max(range, (double)fabs(a.data()[i]));
      worst = std::max(worst, fabs((double)a.data()[i] - (double)b.data()[i]));
    }
    // Per-channel int8 keeps the error well below the output range.
    CHECK(worst < 0.05 * range + 1e-3);
  }
}

void testModelCalibratedInt8Activations() {
  NetworkDescription desc;
  BufferStream cfg(kMiniCfg, sizeof(kMiniCfg) - 1);
  CHECK(desc.parseCfg(cfg) == Status::Ok);
  ReferenceBackend be;
  Model float_model, int8_model;
  CHECK(float_model.build(desc.spec(), be, WeightPrecision::Float32) == Status::Ok);
  CHECK(int8_model.build(desc.spec(), be, WeightPrecision::Int8) == Status::Ok);

  std::vector<float> weights(float_model.paramCount());
  unsigned seed = 7;
  for (float& w : weights) w = unitRandom(seed) * 0.3f;
  ArrayWeightSource s1(weights.data(), weights.size());
  ArrayWeightSource s2(weights.data(), weights.size());
  CHECK(float_model.loadWeights(s1) == Status::Ok);
  CHECK(int8_model.loadWeights(s2) == Status::Ok);

  Tensor sample;
  CHECK(sample.allocate(int8_model.inputShape()) == Status::Ok);
  for (size_t i = 0; i < sample.size(); ++i) sample.data()[i] = 0.5f + 0.5f * unitRandom(seed);

  CHECK(int8_model.int8ActivationLayers() == 0);
  CHECK(int8_model.calibrate(sample) == Status::Ok);
  CHECK(int8_model.int8ActivationLayers() > 0);

  // The float32 model has nothing to calibrate: its convolutions keep float activations.
  CHECK(float_model.calibrate(sample) == Status::Ok);
  CHECK(float_model.int8ActivationLayers() == 0);

  Tensor wrong;
  CHECK(wrong.allocate(Shape(1, 1, 1)) == Status::Ok);
  CHECK(int8_model.calibrate(wrong) == Status::ShapeMismatch);

  for (size_t i = 0; i < sample.size(); ++i) {
    float_model.input().data()[i] = sample.data()[i];
    int8_model.input().data()[i] = sample.data()[i];
  }
  CHECK(float_model.run() == Status::Ok);
  CHECK(int8_model.run() == Status::Ok);
  for (int h = 0; h < float_model.headCount(); ++h) {
    const Tensor& a = float_model.head(h).output();
    const Tensor& b = int8_model.head(h).output();
    double range = 0.0, worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      range = std::max(range, (double)fabs(a.data()[i]));
      worst = std::max(worst, fabs((double)a.data()[i] - (double)b.data()[i]));
    }
    // Int8 weights and int8 activations together: a few percent of the output range.
    CHECK(worst < 0.1 * range + 1e-3);
  }
}

void testDetectorWithInt8Weights() {
  // The probe network with int8 weights: the same single box as with float weights.
  std::vector<float> weights = probeWeights();
  ArrayWeightSource source(weights.data(), weights.size());
  DetectorConfig config;
  config.weight_precision = WeightPrecision::Int8;
  Detector detector;
  CHECK(detector.begin(kProbeNet, source, config) == Status::Ok);
  std::vector<uint8_t> pixels;
  makeRedImage(pixels, {{3, 5}});
  ImageView image(pixels.data(), 8, 8, PixelFormat::RGB888);
  Detections out;
  out.reserve(4);
  CHECK(detector.detect(image, out) == Status::Ok);
  CHECK(out.size() == 1);
  if (out.size() == 1) {
    CHECK_NEAR(out[0].x, 1.5f, 1e-3);
    CHECK_NEAR(out[0].width, 4.0f, 1e-3);
  }
}

void testPerLayerPrecision() {
  // Probe network: conv(6) feeds the YOLO head directly, so it is a head feeder.
  ReferenceBackend be;
  const size_t float_conv_bytes = (18 + 6) * sizeof(float);
  const size_t int8_conv_bytes = 18 * sizeof(int8_t) + 6 * sizeof(float) /*scales*/ + 6 * sizeof(float) /*bias*/;

  // Int8 model: the head feeder keeps float32 weights.
  Model int8_model;
  CHECK(int8_model.build(kProbeNet, be, WeightPrecision::Int8) == Status::Ok);
  CHECK(int8_model.layer(0).weightBytes() == float_conv_bytes);

  // Float model: nothing changes.
  Model float_model;
  CHECK(float_model.build(kProbeNet, be, WeightPrecision::Float32) == Status::Ok);
  CHECK(float_model.layer(0).weightBytes() == float_conv_bytes);

  // Per-layer override in an int8 model: force int8 on the head feeder.
  static const LayerSpec forced_int8[] = {
      LayerSpec::conv(6, 1, 1, Activation::Linear).withPrecision(WeightPrecision::Int8),
      LayerSpec::yolo(1, kAnchor, 1),
  };
  Model forced;
  CHECK(forced.build({8, 8, 3, forced_int8, 2, nullptr, 0}, be, WeightPrecision::Int8) == Status::Ok);
  CHECK(forced.layer(0).weightBytes() == int8_conv_bytes);

  // Per-layer override in a float model: force float32 on a layer that would otherwise be int8.
  static const LayerSpec forced_float[] = {
      LayerSpec::conv(6, 1, 1, Activation::Linear).withPrecision(WeightPrecision::Float32),
      LayerSpec::conv(6, 1, 1, Activation::Linear),
      LayerSpec::yolo(1, kAnchor, 1),
  };
  Model mixed;
  CHECK(mixed.build({8, 8, 3, forced_float, 3, nullptr, 0}, be, WeightPrecision::Int8) == Status::Ok);
  // Layer 0 is overridden to float32; layer 1 feeds the head, so it is float32 too.
  CHECK(mixed.layer(0).weightBytes() == float_conv_bytes);
  // Layer 1 reads 6 channels: 6x6 weights plus 6 biases, all float32.
  CHECK(mixed.layer(1).weightBytes() == (36 + 6) * sizeof(float));
}

void testActivationReuseKeepsResults() {
  // The planned model (outputs share memory) must give exactly the same results as one where
  // every output has its own memory.
  NetworkDescription desc;
  BufferStream cfg(kMiniCfg, sizeof(kMiniCfg) - 1);
  CHECK(desc.parseCfg(cfg) == Status::Ok);
  ReferenceBackend be;
  Model shared, separate;
  CHECK(shared.build(desc.spec(), be, WeightPrecision::Float32, true) == Status::Ok);
  CHECK(separate.build(desc.spec(), be, WeightPrecision::Float32, false) == Status::Ok);
  CHECK(shared.activationBytes() < separate.activationBytes());
  CHECK(separate.activationBytes() == separate.activationBytesWithoutReuse());
  CHECK(shared.activationBytesWithoutReuse() == separate.activationBytes());

  std::vector<float> weights(shared.paramCount());
  unsigned seed = 11;
  for (float& w : weights) w = unitRandom(seed) * 0.3f;
  ArrayWeightSource s1(weights.data(), weights.size());
  ArrayWeightSource s2(weights.data(), weights.size());
  CHECK(shared.loadWeights(s1) == Status::Ok);
  CHECK(separate.loadWeights(s2) == Status::Ok);

  for (int round = 0; round < 3; ++round) {  // several frames, so stale data would show
    for (size_t i = 0; i < shared.input().size(); ++i) {
      const float v = 0.5f + 0.5f * unitRandom(seed);
      shared.input().data()[i] = v;
      separate.input().data()[i] = v;
    }
    CHECK(shared.run() == Status::Ok);
    CHECK(separate.run() == Status::Ok);
    for (int h = 0; h < shared.headCount(); ++h) {
      const Tensor& a = shared.head(h).output();
      const Tensor& b = separate.head(h).output();
      CHECK(a.shape() == b.shape());
      bool identical = true;
      for (size_t i = 0; i < a.size(); ++i) identical = identical && a.data()[i] == b.data()[i];
      CHECK(identical);
    }
  }
}

void testMishAndShortcut() {
  // Mish: x * tanh(softplus(x)). Known values: mish(0) = 0, mish(1) ~ 0.8651.
  CHECK_NEAR(activate(Activation::Mish, 0.0f), 0.0f, 1e-6);
  CHECK_NEAR(activate(Activation::Mish, 1.0f), 0.8651f, 1e-3);
  CHECK_NEAR(activate(Activation::Mish, 30.0f), 30.0f, 1e-3);

  // Residual add: a 2x2x1 tensor plus a 2x2x1 tensor, with leaky activation.
  ReferenceBackend be;
  Tensor a, b, out;
  CHECK(a.allocate(Shape(2, 2, 1)) == Status::Ok);
  CHECK(b.allocate(Shape(2, 2, 1)) == Status::Ok);
  CHECK(out.allocate(Shape(2, 2, 1)) == Status::Ok);
  a.fill(1.0f);
  b.fill(-3.0f);
  CHECK(be.add(a, b, Activation::Linear, out) == Status::Ok);
  CHECK_NEAR(out.at(1, 1, 0), -2.0f, 1e-6);
  CHECK(be.add(a, b, Activation::Leaky, out) == Status::Ok);
  CHECK_NEAR(out.at(0, 0, 0), -0.2f, 1e-6);

  // A Darknet [shortcut] becomes an Add layer; the model runs and the shapes match.
  static const char kShortcutCfg[] =
      "[net]\nwidth=8\nheight=8\nchannels=3\n"
      "[convolutional]\nfilters=4\nsize=3\nstride=1\npad=1\nactivation=linear\n"
      "[convolutional]\nfilters=4\nsize=3\nstride=1\npad=1\nactivation=linear\n"
      "[shortcut]\nfrom=-2\nactivation=linear\n"
      "[convolutional]\nfilters=6\nsize=1\nstride=1\npad=0\nactivation=linear\n"
      "[yolo]\nmask=0\nanchors=4,4\nclasses=1\n";
  BufferStream cfg_stream(kShortcutCfg, sizeof(kShortcutCfg) - 1);
  NetworkDescription desc;
  CHECK(desc.parseCfg(cfg_stream) == Status::Ok);
  CHECK(desc.spec().layers[2].type == LayerType::Add);
  CHECK(desc.spec().layers[2].from[1] == 0);  // -3 relative to layer 2 is layer 0
  Model model;
  CHECK(model.build(desc.spec(), be) == Status::Ok);
  CHECK(model.layer(2).outputShape() == Shape(8, 8, 4));
  std::vector<float> weights(model.paramCount(), 0.05f);
  ArrayWeightSource source(weights.data(), weights.size());
  CHECK(model.loadWeights(source) == Status::Ok);
  model.input().fill(0.5f);
  CHECK(model.run() == Status::Ok);
}

void testShortcutCfgRejectsMismatch() {
  // A shortcut whose inputs differ in size is rejected at build time.
  static const char kBad[] =
      "[net]\nwidth=8\nheight=8\nchannels=3\n"
      "[convolutional]\nfilters=4\nsize=3\nstride=1\npad=1\nactivation=linear\n"
      "[convolutional]\nfilters=4\nsize=3\nstride=2\npad=1\nactivation=linear\n"
      "[shortcut]\nfrom=-2\nactivation=linear\n";
  BufferStream stream(kBad, sizeof(kBad) - 1);
  NetworkDescription desc;
  CHECK(desc.parseCfg(stream) == Status::Ok);
  ReferenceBackend be;
  Model model;
  ScopedLog log(TinyYoloLogLevel::Error);
  CHECK(model.build(desc.spec(), be) == Status::ShapeMismatch);
  CHECK(contains(log.out.text, "[E] Add: shortcut inputs differ"));
}

// ---------------------------------------------------------------- App layer

/// A ModelSource over three in-memory buffers, like FlashModelSource but with any size.
class BufferModelSource : public ModelSource {
 public:
  BufferModelSource(const char* cfg, const char* names, const std::vector<uint8_t>& weights)
      : cfg_(cfg, strlen(cfg)), names_(names, strlen(names)), weights_(weights.data(), weights.size()) {}
  bool open() override {
    cfg_.rewind();
    names_.rewind();
    weights_.rewind();
    return true;
  }
  void close() override {}
  Stream& cfg() override { return cfg_; }
  Stream& names() override { return names_; }
  Stream& weights() override { return weights_; }

 private:
  BufferStream cfg_;
  BufferStream names_;
  BufferStream weights_;
};

/// Darknet v0.2 weight file for the probe network (non-batch-norm: bias first, then weights).
std::vector<uint8_t> probeDarknetWeights() {
  std::vector<uint8_t> file;
  const uint32_t version[3] = {0, 2, 0};
  const uint64_t seen = 0;
  appendBytes(file, version, sizeof(version));
  appendBytes(file, &seen, sizeof(seen));
  std::vector<float> w = probeWeights();  // weights (18) then biases (6)
  std::vector<float> body(w.begin() + 18, w.end());
  body.insert(body.end(), w.begin(), w.begin() + 18);
  appendBytes(file, body.data(), body.size() * sizeof(float));
  return file;
}

const char kProbeCfg[] =
    "[net]\nwidth=8\nheight=8\nchannels=3\n"
    "[convolutional]\nfilters=6\nsize=1\nstride=1\npad=0\nactivation=linear\n"
    "[yolo]\nmask=0\nanchors=4,4\nclasses=1\n";

void testTinyYoloModelDetects() {
  std::vector<uint8_t> weights = probeDarknetWeights();
  BufferModelSource source(kProbeCfg, "bright\n", weights);
  TinyYoloModel model;
  CHECK(!model.isReady());
  CHECK(model.begin(source) == Status::Ok);
  CHECK(model.isReady());
  CHECK(model.inputWidth() == 8 && model.inputHeight() == 8);
  CHECK(strcmp(model.label(0), "bright") == 0);

  std::vector<uint8_t> pixels;
  makeRedImage(pixels, {{3, 5}});
  ImageView image(pixels.data(), 8, 8, PixelFormat::RGB888);
  CHECK(model.detect(image) == Status::Ok);
  CHECK(model.detections().size() == 1);
  if (model.detections().size() == 1) CHECK_NEAR(model.detections()[0].x, 1.5f, 1e-3);

  CapturePrint out;
  Output output(out);
  for (const Detection& d : model.detections()) {
    output.printf("%s %.2f at %d,%d %dx%d", model.detector().label(d.class_id), d.score,
                   (int)d.x, (int)d.y, (int)d.width, (int)d.height);
  }
  CHECK(contains(out.text, "bright 1.00 at 1,3 4x4"));
}

void testTinyYoloModelFailures() {
  // Truncated weights: begin() fails, the model is not ready, and detect() reports it.
  std::vector<uint8_t> weights = probeDarknetWeights();
  weights.resize(weights.size() - 8);
  BufferModelSource source(kProbeCfg, "bright\n", weights);
  TinyYoloModel model;
  ScopedLog log(TinyYoloLogLevel::Error);
  CHECK(model.begin(source) != Status::Ok);
  CHECK(!model.isReady());
  CHECK(contains(log.out.text, "[E] TinyYoloModel: setup failed"));

  std::vector<uint8_t> pixels(8 * 8 * 3, 0);
  ImageView image(pixels.data(), 8, 8, PixelFormat::RGB888);
  CHECK(model.detect(image) == Status::NotInitialized);
}

void testSyntheticFrame() {
  SyntheticFrame frames(8, 8);
  ImageView first = frames.next();
  CHECK(first.width() == 8 && first.height() == 8 && first.format() == PixelFormat::RGB888);
  float before[3];
  first.rgbAt(0, 2, before);  // inside the square in frame 1
  ImageView second = frames.next();
  float after[3];
  second.rgbAt(0, 2, after);  // the square has moved away from x = 0
  CHECK(before[0] > 0.9f && after[0] < 0.1f);
  CHECK(frames.count() == 2);
}

void testOutputPrintf() {
  // Output formats like printf and adds the line end; with no detections nothing is printed.
  CapturePrint out;
  Output output(out);
  output.printf("score %.2f at %d,%d", 0.5, 3, 4);
  CHECK(contains(out.text, "score 0.50 at 3,4"));
  CHECK(!out.text.empty() && out.text.back() == '\n');
  CapturePrint none;
  Output quiet(none);
  Detections empty;
  for (const Detection& d : empty) quiet.printf("%d", d.class_id);
  CHECK(none.text.empty());
}

void testModelBlobRoundTrip() {
  // An int8 model is written to a blob and loaded again. The loaded model reads its int8
  // weights from the blob's memory (no copy) and must give exactly the same outputs.
  NetworkDescription desc;
  BufferStream cfg(kMiniCfg, sizeof(kMiniCfg) - 1);
  CHECK(desc.parseCfg(cfg) == Status::Ok);
  ReferenceBackend be;

  Model original;
  CHECK(original.build(desc.spec(), be, WeightPrecision::Int8) == Status::Ok);
  std::vector<float> weights(original.paramCount());
  unsigned seed = 21;
  for (float& w : weights) w = unitRandom(seed) * 0.3f;
  ArrayWeightSource source_a(weights.data(), weights.size());
  CHECK(original.loadWeights(source_a) == Status::Ok);

  struct VectorSink {
    std::vector<uint8_t> bytes;
    size_t write(const void* data, size_t n) {
      const uint8_t* p = static_cast<const uint8_t*>(data);
      bytes.insert(bytes.end(), p, p + n);
      return n;
    }
  } sink;
  CHECK(writeModelBlob(original, sink) == Status::Ok);
  CHECK(sink.bytes.size() > 8);

  // The blob is loaded into a model that keeps views of it.
  ModelBlobSource blob_source(sink.bytes.data(), sink.bytes.size());
  CHECK(blob_source.open() == Status::Ok);
  Model loaded;
  CHECK(loaded.build(desc.spec(), be, WeightPrecision::Int8) == Status::Ok);
  CHECK(loaded.loadWeights(blob_source) == Status::Ok);
  const ConvLayer* first_int8 = nullptr;
  for (int i = 0; i < loaded.layerCount() && !first_int8; ++i) {
    const ConvLayer* c = loaded.layer(i).asConv();
    if (c != nullptr && c->precision() == WeightPrecision::Int8) first_int8 = c;
  }
  CHECK(first_int8 != nullptr);
  if (first_int8 != nullptr) {
    const uint8_t* lo = sink.bytes.data();
    const uint8_t* hi = lo + sink.bytes.size();
    const uint8_t* at = reinterpret_cast<const uint8_t*>(first_int8->int8Weights());
    CHECK(at >= lo && at < hi);  // a view into the blob, not a copy
    CHECK(reinterpret_cast<uintptr_t>(first_int8->scales()) % 4 == 0);
  }

  for (size_t i = 0; i < original.input().size(); ++i) {
    const float v = 0.5f + 0.5f * unitRandom(seed);
    original.input().data()[i] = v;
    loaded.input().data()[i] = v;
  }
  CHECK(original.run() == Status::Ok);
  CHECK(loaded.run() == Status::Ok);
  for (int h = 0; h < original.headCount(); ++h) {
    const Tensor& a = original.head(h).output();
    const Tensor& b = loaded.head(h).output();
    bool identical = a.shape() == b.shape();
    for (size_t i = 0; identical && i < a.size(); ++i) identical = a.data()[i] == b.data()[i];
    CHECK(identical);
  }

  // A float model reads the same blob, dequantized: it must build, run and stay finite.
  ModelBlobSource float_source(sink.bytes.data(), sink.bytes.size());
  CHECK(float_source.open() == Status::Ok);
  Model as_float;
  CHECK(as_float.build(desc.spec(), be, WeightPrecision::Float32) == Status::Ok);
  CHECK(as_float.loadWeights(float_source) == Status::Ok);
  as_float.input().fill(0.5f);
  CHECK(as_float.run() == Status::Ok);

  // Damaged or foreign data is rejected.
  std::vector<uint8_t> junk(64, 7);
  ModelBlobSource bad(junk.data(), junk.size());
  ScopedLog log(TinyYoloLogLevel::Error);
  CHECK(bad.open() == Status::WeightError);
  CHECK(contains(log.out.text, "not a model blob"));
}

void testQuantizedConvAndPool() {
  const int H = 8, W = 8, IC = 3, OC = 4, K = 3;
  Tensor in;
  CHECK(in.allocate(Shape(H, W, IC)) == Status::Ok);
  for (size_t i = 0; i < in.size(); ++i) in.data()[i] = sinf(0.37f * (float)i);
  std::vector<float> w(OC * IC * K * K);
  for (size_t i = 0; i < w.size(); ++i) w[i] = 0.5f * cosf(0.91f * (float)i);
  const float bias[OC] = {0.1f, -0.2f, 0.0f, 0.05f};
  const ConvParams p = {K, 1, 1, OC, Activation::Linear};

  ReferenceBackend be;
  Tensor ref;
  CHECK(ref.allocate(Shape(H, W, OC)) == Status::Ok);
  CHECK(be.conv2d(in, w.data(), bias, p, ref) == Status::Ok);

  // Quantized input, int8 weights (per output channel) and int8 output.
  const float s_in = maxAbsValue(in) / 127.0f;
  QTensor qin;
  CHECK(qin.allocate(in.shape(), s_in) == Status::Ok);
  CHECK(quantizeTensor(in, s_in, qin) == Status::Ok);
  std::vector<int8_t> qw(w.size());
  std::vector<float> ws(OC);
  quantizeRows(w.data(), OC, IC * K * K, qw.data(), ws.data());
  qw = toOhwi(qw, OC, IC, K);  // the int8 kernels take OHWI
  const float s_out = maxAbsValue(ref) / 127.0f;
  QTensor qout;
  CHECK(qout.allocate(ref.shape(), s_out) == Status::Ok);
  CHECK(be.conv2dQ(qin, qw.data(), ws.data(), bias, p, qout) == Status::Ok);
  CHECK(qout.scale() == s_out);

  Tensor back;
  CHECK(back.allocate(ref.shape()) == Status::Ok);
  CHECK(dequantizeTensor(qout, back) == Status::Ok);
  float max_err = 0.0f;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float e = fabsf(back.data()[i] - ref.data()[i]);
    if (e > max_err) max_err = e;
  }
  // Rounding of input, weights and output: a few percent of the value range, plus half an
  // output step.
  CHECK(max_err <= 0.05f * maxAbsValue(ref) + s_out);

  // Shape and scale errors are reported, not silently accepted.
  QTensor wrong;
  CHECK(wrong.allocate(Shape(H, W, OC + 1), s_out) == Status::Ok);
  CHECK(be.conv2dQ(qin, qw.data(), ws.data(), bias, p, wrong) == Status::ShapeMismatch);
  QTensor none;
  CHECK(none.allocate(Shape(H, W, OC), 1.0f) == Status::Ok);
  none.setScale(0.0f);
  CHECK(be.conv2dQ(qin, qw.data(), ws.data(), bias, p, none) == Status::InvalidArgument);
  QTensor zero_scale;
  CHECK(zero_scale.allocate(Shape(H, W, OC), 0.0f) == Status::InvalidArgument);

  // ReLU: no negative value survives the int8 path.
  const ConvParams relu = {K, 1, 1, OC, Activation::ReLU};
  QTensor qrelu;
  CHECK(qrelu.allocate(ref.shape(), s_out) == Status::Ok);
  CHECK(be.conv2dQ(qin, qw.data(), ws.data(), bias, relu, qrelu) == Status::Ok);
  for (size_t i = 0; i < qrelu.size(); ++i) CHECK(qrelu.data()[i] >= 0);

  // Max pooling on int8 equals quantizing the float max pooling of the same values exactly
  // (rounding is monotonic and the scale is shared).
  const int pooled_h = poolOutputSize(H, 2, 2, 0);
  const int pooled_w = poolOutputSize(W, 2, 2, 0);
  Tensor pooled_f;
  CHECK(pooled_f.allocate(Shape(pooled_h, pooled_w, OC)) == Status::Ok);
  CHECK(be.maxPool(back, 2, 2, 0, pooled_f) == Status::Ok);
  QTensor expected;
  CHECK(expected.allocate(pooled_f.shape(), s_out) == Status::Ok);
  CHECK(quantizeTensor(pooled_f, s_out, expected) == Status::Ok);
  QTensor pooled_q;
  CHECK(pooled_q.allocate(pooled_f.shape(), s_out) == Status::Ok);
  CHECK(be.maxPoolQ(qout, 2, 2, 0, pooled_q) == Status::Ok);
  bool same = true;
  for (size_t i = 0; i < pooled_q.size(); ++i) same = same && pooled_q.data()[i] == expected.data()[i];
  CHECK(same);
}

void testEsp32NnFallbackMatchesReference() {
  // Off the ESP32-S3 the Esp32NnBackend int8 convolution must be the portable kernel exactly.
  CHECK(Esp32NnBackend::hasEspNn() == false);
  Esp32NnBackend nn(1);
  CHECK(strcmp(nn.name(), "esp32-nn") != 0);
  const int H = 6, W = 5, IC = 4, OC = 8, K = 3, S = 1, P = 1;
  Tensor in_f;
  CHECK(in_f.allocate(Shape(H, W, IC)) == Status::Ok);
  for (size_t i = 0; i < in_f.size(); ++i) in_f.data()[i] = sinf(0.7f * (float)i);
  QTensor in;
  CHECK(in.allocate(in_f.shape(), 0.01f) == Status::Ok);
  CHECK(quantizeTensor(in_f, 0.01f, in) == Status::Ok);
  std::vector<float> w(OC * IC * K * K);
  for (size_t i = 0; i < w.size(); ++i) w[i] = 0.4f * cosf(0.3f * (float)i);
  std::vector<int8_t> qw(w.size());
  std::vector<float> ws(OC);
  quantizeRows(w.data(), OC, IC * K * K, qw.data(), ws.data());
  qw = toOhwi(qw, OC, IC, K);
  std::vector<float> bias(OC, 0.05f);
  const ConvParams p = {K, S, P, OC, Activation::Leaky};
  const int oh = windowOutputSize(H, K, S, P), ow = windowOutputSize(W, K, S, P);
  QTensor a, b;
  CHECK(a.allocate(Shape(oh, ow, OC), 0.02f) == Status::Ok);
  CHECK(b.allocate(Shape(oh, ow, OC), 0.02f) == Status::Ok);
  ReferenceBackend reference;
  CHECK(reference.conv2dQ(in, qw.data(), ws.data(), bias.data(), p, a) == Status::Ok);
  CHECK(nn.conv2dQ(in, qw.data(), ws.data(), bias.data(), p, b) == Status::Ok);
  bool same = true;
  for (size_t i = 0; i < a.size(); ++i) same = same && a.data()[i] == b.data()[i];
  CHECK(same);
}

void testImageScaleTo() {
  // A 4x4 RGB565 image (big-endian bytes) with a distinct value per pixel.
  uint8_t src[4 * 4 * 2];
  for (int i = 0; i < 16; ++i) {
    const uint16_t v = (uint16_t)(0x1000 + i * 0x0111);
    src[2 * i] = (uint8_t)(v >> 8);
    src[2 * i + 1] = (uint8_t)(v & 0xFF);
  }
  ImageView image(src, 4, 4, PixelFormat::RGB565Big);
  uint8_t dst[2 * 2 * 2];
  const ImageView small = image.scaleTo(dst, 2, 2, PixelFormat::RGB565Little);
  CHECK(small.data() == dst);
  CHECK(small.byteCount() == 8);
  CHECK(small.width() == 2 && small.height() == 2);
  // Destination pixel (x, y) samples source (2x + 1, 2y + 1); output bytes are little-endian.
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 2; ++x) {
      const int i = (2 * y + 1) * 4 + (2 * x + 1);
      const uint16_t v = (uint16_t)(0x1000 + i * 0x0111);
      const uint8_t* d = dst + (y * 2 + x) * 2;
      CHECK(d[0] == (uint8_t)(v & 0xFF));
      CHECK(d[1] == (uint8_t)(v >> 8));
    }
  }
  // Same image into big-endian output: the bytes are the source bytes at the sampled pixels.
  CHECK(image.scaleTo(dst, 2, 2, PixelFormat::RGB565Big).data() == dst);
  CHECK(dst[0] == src[2 * (1 * 4 + 1)] && dst[1] == src[2 * (1 * 4 + 1) + 1]);
  // Unsupported target and missing destination give an empty view.
  CHECK(image.scaleTo(dst, 2, 2, PixelFormat::RGB888).data() == nullptr);
  CHECK(image.scaleTo(nullptr, 2, 2, PixelFormat::RGB565Big).data() == nullptr);
  CHECK(image.scaleTo(nullptr, 2, 2, PixelFormat::RGB565Big).byteCount() == 0);
  // A 24-bit source converts through rgbAt(): pure red becomes 0xF800 (big-endian bytes F8 00).
  const uint8_t red[3] = {255, 0, 0};
  ImageView rgb(red, 1, 1, PixelFormat::RGB888);
  uint8_t out[2];
  CHECK(rgb.scaleTo(out, 1, 1, PixelFormat::RGB565Big).data() == out);
  CHECK(out[0] == 0xF8 && out[1] == 0x00);
}

}  // namespace

// ---- P4 convolution schedule and attention blocks -------------------------------------------
// The ESP32-P4 assembly kernels only run on the chip. The schedule (ConvSchedule.h) takes the
// kernel as a parameter, so these tests run the real schedule with a host stand-in that reads
// the same argument block, and compare with a naive convolution.

static int g_mock_taps = 9;

/// Host stand-in for one P4 kernel call: one output pixel, 16 channels. It reads the argument
/// block the way the assembly does (filter, bias, shift or per-channel shifts, tap steps).
static void mockConvKernel(int8_t* output, int8_t* input, void* argp) {
  const uint32_t* w = static_cast<const uint32_t*>(argp);
  const esp32p4::ConvArgs& a = *static_cast<const esp32p4::ConvArgs*>(argp);
  const int8_t* filter = static_cast<const int8_t*>(a.filter_ptr);
  const int32_t* bias = static_cast<const int32_t*>(a.bias_ptr);
  const int8_t* shifts = static_cast<const int8_t*>(a.shift_ptr);
  const bool per_channel = w[esp32p4::ConvArgs::kMacShift] == 0x80000000u;
  const int C = ((int)w[esp32p4::ConvArgs::kInputGroupsM1] + 1) * 16;
  const int taps = g_mock_taps;
  const int row = 3 * C + (int)w[esp32p4::ConvArgs::kDilationY];  // the padded row length
  for (int j = 0; j < 16; ++j) {
    int32_t acc = bias[j];
    for (int t = 0; t < taps; ++t) {
      const int off = taps == 9 ? (t / 3) * row + (t % 3) * C : 0;
      for (int c = 0; c < C; ++c) acc += (int32_t)input[off + c] * (int32_t)filter[(t * C + c) * 16 + j];
    }
    const int shift = per_channel ? (int)shifts[j] : (int)w[esp32p4::ConvArgs::kMacShift];
    const int32_t v = acc >> shift;
    output[j] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
  }
}

static uint32_t g_rand_state = 12345u;
static int randInRange(int lo, int hi) {
  g_rand_state = g_rand_state * 1664525u + 1013904223u;
  return lo + (int)((g_rand_state >> 8) % (uint32_t)(hi - lo + 1));
}

/// Storage with a 16-byte aligned start, as the kernels need.
struct AlignedInt8 {
  std::vector<int8_t> store;
  int8_t* p;
  explicit AlignedInt8(size_t n) : store(n + 16, 0) {
    p = store.data() + ((16 - ((uintptr_t)store.data() & 15)) & 15);
  }
};

/// Naive reference: valid convolution of the padded input (HWC) with OHWI weights.
static std::vector<int8_t> naiveConvInt8(const std::vector<int8_t>& in, int Wp, int C, int K, int S, int OH,
                                         int OW, int OC, const std::vector<int8_t>& w, const std::vector<int32_t>& bias,
                                         int mac_shift, const std::vector<int8_t>* channel_shifts) {
  std::vector<int8_t> out((size_t)OH * OW * OC);
  for (int oy = 0; oy < OH; ++oy) {
    for (int ox = 0; ox < OW; ++ox) {
      for (int oc = 0; oc < OC; ++oc) {
        int32_t acc = bias[(size_t)oc];
        for (int ky = 0; ky < K; ++ky) {
          for (int kx = 0; kx < K; ++kx) {
            for (int c = 0; c < C; ++c) {
              const size_t i = ((size_t)(oy * S + ky) * Wp + (size_t)(ox * S + kx)) * C + c;
              acc += (int32_t)in[i] * (int32_t)w[(((size_t)oc * K + ky) * K + kx) * C + c];
            }
          }
        }
        const int shift = channel_shifts ? (int)(*channel_shifts)[(size_t)oc] : mac_shift;
        const int32_t v = acc >> shift;
        out[((size_t)oy * OW + ox) * OC + oc] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
      }
    }
  }
  return out;
}

void testConvBlocksMatchNaiveReference() {
  struct Case {
    int K, S, C, OC, H, W;
  };
  // Whole filter in the staging buffer, block copies, stride 2, 1x1, the largest block (C = 256,
  // whose 64-byte tail ends exactly at the buffer end), and a single output row.
  const Case cases[] = {
      {3, 1, 16, 16, 9, 7}, {3, 1, 32, 64, 10, 8}, {3, 2, 64, 32, 11, 9}, {1, 1, 32, 48, 6, 5},
      {1, 1, 256, 64, 4, 4}, {3, 1, 256, 16, 6, 6}, {3, 1, 16, 16, 1, 1},
  };
  for (const Case& c : cases) {
    const int pad = c.K == 3 ? 1 : 0;
    const int Hp = c.H + 2 * pad, Wp = c.W + 2 * pad;
    const int OH = (Hp - c.K) / c.S + 1, OW = (Wp - c.K) / c.S + 1;
    const int taps = c.K * c.K;
    // Input with a zero border (the caller writes it), OHWI weights, and the packing of the kernels.
    std::vector<int8_t> in((size_t)Hp * Wp * c.C, 0);
    for (int y = 0; y < c.H; ++y)
      for (int x = 0; x < c.W; ++x)
        for (int ch = 0; ch < c.C; ++ch)
          in[((size_t)(y + pad) * Wp + (x + pad)) * c.C + ch] = (int8_t)randInRange(-127, 127);
    std::vector<int8_t> w((size_t)c.OC * taps * c.C);
    for (auto& v : w) v = (int8_t)randInRange(-127, 127);
    AlignedInt8 packed((size_t)c.OC * taps * c.C);
    for (int oc = 0; oc < c.OC; ++oc)
      for (int t = 0; t < taps; ++t)
        for (int ic = 0; ic < c.C; ++ic)
          packed.p[(((oc / 16) * taps + t) * c.C + ic) * 16 + oc % 16] = w[((size_t)oc * taps + t) * c.C + ic];
    std::vector<int32_t> bias((size_t)c.OC);
    for (auto& b : bias) b = randInRange(-500, 500);
    std::vector<int8_t> shifts((size_t)c.OC);
    for (auto& s : shifts) s = (int8_t)randInRange(4, 9);
    // The kernel gets the unpadded picture; the reference above uses the padded copy.
    std::vector<int8_t> unpadded((size_t)c.H * c.W * c.C);
    for (int y = 0; y < c.H; ++y)
      std::copy(in.begin() + (((size_t)(y + pad) * Wp + pad) * c.C),
                in.begin() + (((size_t)(y + pad) * Wp + pad + c.W) * c.C),
                unpadded.begin() + (size_t)y * c.W * c.C);
    AlignedInt8 inBuf(unpadded.size());
    std::copy(unpadded.begin(), unpadded.end(), inBuf.p);
    AlignedInt8 shiftBuf((size_t)c.OC);
    std::copy(shifts.begin(), shifts.end(), shiftBuf.p);
    const int mac_shift = 6;

    for (int per_channel = 0; per_channel < 2; ++per_channel) {
      const std::vector<int8_t> expected = naiveConvInt8(in, Wp, c.C, c.K, c.S, OH, OW, c.OC, w, bias, mac_shift,
                                                          per_channel ? &shifts : nullptr);
      for (int workers : {1, 2, 4}) {
        AlignedInt8 out(expected.size());
        g_mock_taps = taps;
        const Status s = esp32p4::runConvBlocks(&mockConvKernel, c.K == 3, c.C, c.OC, c.H, c.W, pad, c.S, OH, OW, inBuf.p,
                                                packed.p, mac_shift, bias.data(),
                                                per_channel ? shiftBuf.p : nullptr, out.p, workers);
        CHECK(s == Status::Ok);
        bool same = true;
        for (size_t i = 0; i < expected.size(); ++i) same = same && out.p[i] == expected[i];
        CHECK(same);
      }
    }
  }
}

/// Host stand-in for the P4 depthwise kernel: one pixel, all channels. It walks the taps with the
/// pointer steps of the argument block (x step, row step, and the step from the last tap of a
/// channel group to the next group), as the assembly does.
static void mockDepthwiseKernel(int8_t* output, int8_t* input, void* argp) {
  const esp32p4::ConvArgs& a = *static_cast<const esp32p4::ConvArgs*>(argp);
  const uint32_t* w = a.words;
  const int8_t* packed = static_cast<const int8_t*>(a.filter_ptr);
  const int32_t* bias = static_cast<const int32_t*>(a.bias_ptr);
  const int groups = (int)w[esp32p4::ConvArgs::kInputGroupsM1] + 1;
  const int dil_x = (int)w[esp32p4::ConvArgs::kDepthDilationX];
  const int row = (int)w[esp32p4::ConvArgs::kDepthDilationY] + 2 * dil_x;  // taps of one row, then the row step
  const int next = (int)w[esp32p4::ConvArgs::kDepthNextGroup];
  const int shift = (int)w[esp32p4::ConvArgs::kMacShift];
  const int8_t* s = input;  // first tap of the group
  for (int g = 0; g < groups; ++g) {
    for (int j = 0; j < 16; ++j) {
      const int c = 16 * g + j;
      int32_t acc = bias[c];
      for (int ty = 0; ty < 3; ++ty) {
        for (int tx = 0; tx < 3; ++tx) {
          const int t = ty * 3 + tx;
          acc += (int32_t)s[ty * row + tx * dil_x + j] * (int32_t)packed[(g * 9 + t) * 16 + j];
        }
      }
      const int32_t v = acc >> shift;
      output[c] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
    }
    s = s + 2 * row + 2 * dil_x + next;  // from the last tap of this group to the first of the next
  }
}

void testDepthwiseBlocksMatchNaiveReference() {
  struct Case {
    int C, H, W;
  };
  // Interior windows, the picture edges (gathered windows), one-pixel pictures, a 256-channel
  // picture (the largest window), and channel counts that take several groups.
  const Case cases[] = {{16, 1, 1}, {16, 2, 5}, {48, 7, 6}, {256, 5, 4}, {32, 9, 9}, {16, 3, 3}};
  const int shift = 7;
  for (const Case& c : cases) {
    const int Hp = c.H + 2, Wp = c.W + 2;
    std::vector<int8_t> padded((size_t)Hp * Wp * c.C, 0);
    std::vector<int8_t> unpadded((size_t)c.H * c.W * c.C);
    for (int y = 0; y < c.H; ++y)
      for (int x = 0; x < c.W; ++x)
        for (int ch = 0; ch < c.C; ++ch) {
          const int8_t v = (int8_t)randInRange(-127, 127);
          padded[((size_t)(y + 1) * Wp + (x + 1)) * c.C + ch] = v;
          unpadded[((size_t)y * c.W + x) * c.C + ch] = v;
        }
    std::vector<int8_t> w((size_t)c.C * 9);  // [C][3][3], the weight layout of the layer
    for (auto& v : w) v = (int8_t)randInRange(-127, 127);
    // Packed as [C/16][9][16] (depthwise3x3Pack).
    AlignedInt8 packed((size_t)c.C * 9);
    for (int ch = 0; ch < c.C; ++ch)
      for (int t = 0; t < 9; ++t) packed.p[((ch / 16) * 9 + t) * 16 + ch % 16] = w[(size_t)ch * 9 + t];
    AlignedInt8 biasBytes((size_t)c.C * sizeof(int32_t));
    int32_t* bias = reinterpret_cast<int32_t*>(biasBytes.p);
    std::vector<int32_t> bias_ref((size_t)c.C);
    for (int ch = 0; ch < c.C; ++ch) bias_ref[(size_t)ch] = bias[ch] = randInRange(-500, 500);
    AlignedInt8 inBuf(unpadded.size());
    std::copy(unpadded.begin(), unpadded.end(), inBuf.p);

    // Naive reference on the padded picture.
    std::vector<int8_t> expected((size_t)c.H * c.W * c.C);
    for (int y = 0; y < c.H; ++y)
      for (int x = 0; x < c.W; ++x)
        for (int ch = 0; ch < c.C; ++ch) {
          int32_t acc = bias_ref[(size_t)ch];
          for (int ty = 0; ty < 3; ++ty)
            for (int tx = 0; tx < 3; ++tx)
              acc += (int32_t)padded[((size_t)(y + ty) * Wp + (x + tx)) * c.C + ch] * (int32_t)w[(size_t)ch * 9 + ty * 3 + tx];
          const int32_t v = acc >> shift;
          expected[((size_t)y * c.W + x) * c.C + ch] = (int8_t)(v < -128 ? -128 : (v > 127 ? 127 : v));
        }

    for (int workers : {1, 2}) {
      AlignedInt8 out(expected.size());
      const Status s = esp32p4::runDepthwiseBlocks(&mockDepthwiseKernel, c.C, c.H, c.W, inBuf.p, packed.p, shift,
                                                   bias, out.p, workers);
      CHECK(s == Status::Ok);
      bool same = true;
      for (size_t i = 0; i < expected.size(); ++i) same = same && out.p[i] == expected[i];
      CHECK(same);
    }
  }
}

/// Naive attention for one head and query: softmax over the positions, then the weighted values.
static void naiveAttention(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                           int N, int kd, int hd, int C, int h, int nq, float scale, std::vector<float>& y) {
  std::vector<float> s((size_t)N);
  float mx = -1e30f;
  for (int nk = 0; nk < N; ++nk) {
    float dot = 0;
    for (int d = 0; d < kd; ++d) dot += q[((size_t)h * N + nq) * kd + d] * k[((size_t)h * N + nk) * kd + d];
    s[(size_t)nk] = dot * scale;
    mx = std::max(mx, s[(size_t)nk]);
  }
  float sum = 0;
  for (auto& e : s) {
    e = expf(e - mx);
    sum += e;
  }
  for (int d = 0; d < hd; ++d) {
    float a = 0;
    for (int nk = 0; nk < N; ++nk) a += s[(size_t)nk] * v[((size_t)h * N + nk) * hd + d];
    y[(size_t)nq * C + h * hd + d] = a / sum;
  }
}

void testAttentionBlocksMatchSerialAndReference() {
  const int N = 37, heads = 2, kd = 8, hd = 12;  // N is not a multiple of 4; hd is not of 8
  const int C = heads * hd;
  const float scale = 1.0f / sqrtf((float)kd);
  std::vector<float> q((size_t)heads * N * kd), k((size_t)heads * N * kd), v((size_t)heads * N * hd);
  for (auto& x : q) x = 0.5f * cosf(0.37f * (float)(&x - &q[0]));
  for (auto& x : k) x = 0.5f * sinf(0.21f * (float)(&x - &k[0]));
  for (auto& x : v) x = 0.5f * cosf(0.11f * (float)(&x - &v[0]));
  std::vector<float> expected((size_t)N * C, 0.0f);
  for (int h = 0; h < heads; ++h)
    for (int nq = 0; nq < N; ++nq) naiveAttention(q, k, v, N, kd, hd, C, h, nq, scale, expected);

  const int blocks_per_head = (N + 3) / 4;
  const int items = heads * blocks_per_head;
  std::vector<float> serial;
  for (int workers : {1, 2, 4}) {
    const int nw = attentionWorkers(workers, items);
    std::vector<float> scores((size_t)nw * 4 * N, 0.0f);
    std::vector<float> y((size_t)N * C, 0.0f);
    AttentionJob job = {q.data(), k.data(), v.data(), scores.data(), y.data(), N, heads, kd, hd, C, scale,
                        blocks_per_head};
    parallelRange(items, nw, attentionBlocks, &job);
    float max_err = 0.0f;
    for (size_t i = 0; i < y.size(); ++i) max_err = std::max(max_err, fabsf(y[i] - expected[i]));
    CHECK(max_err < 1e-5f);
    if (workers == 1) serial = y;
    bool same_as_serial = true;
    for (size_t i = 0; i < y.size(); ++i) same_as_serial = same_as_serial && y[i] == serial[i];
    CHECK(same_as_serial);
  }
}

int main() {
  testLogger();
  testStatusAndTensor();
  testActivations();
  testConvolution();
  testPoolUpsampleConcat();
  testImageView();
  testLetterbox();
  testDetections();
  testModelBuildAndRun();
  testModelErrors();
  testWeightSources();
  testDetectorDecodesBoxes();
  testDetectorBoundaryClipping();
  testDetectorWarningsAndErrors();
  testLabelCountWarning();
  testFloatReadOverRealisticSize();
  testMaxPoolDarknetStyle();
  testDarknetDecodeStyle();
  testCfgBuildsAndRuns();
  testDarknetBatchNormFolding();
  testDarknetWeightsTruncated();
  testFlashNamesAndBufferStream();
  testDarknetCfgErrors();
  testConvolutionAgreesWithReference();
  testParallelRangeCoversEachIndexOnce();
  testLargeBufferAllocation();
  testDotProductNetworkMatchesReference();
  testTaskRunsAndEnds();
  testDrawRectangle();
  testQuantizeRows();
  testQuantizedConvAndPool();
  testEsp32NnFallbackMatchesReference();
  testImageScaleTo();
  testConvInt8MatchesDequantizedFloat();
  testModelInt8CloseToFloat();
  testModelCalibratedInt8Activations();
  testDetectorWithInt8Weights();
  testPerLayerPrecision();
  testActivationReuseKeepsResults();
  testMishAndShortcut();
  testModelBlobRoundTrip();
  testShortcutCfgRejectsMismatch();
  testTinyYoloModelDetects();
  testTinyYoloModelFailures();
  testSyntheticFrame();
  testOutputPrintf();
  testConvBlocksMatchNaiveReference();
  testAttentionBlocksMatchSerialAndReference();
  testDepthwiseBlocksMatchNaiveReference();

  printf("%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
