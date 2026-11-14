// YOLO26n on the ESP32-P4 with the TinyYolo graph path (tinyyolo::GraphYolo11). The sample photo
// (photo.S, photo.rgb) is detected three times; the weights come from the "model" partition
// (partitions.csv, offset 0x310000). See README.md for the steps.
#include <Arduino.h>
#include <TinyYolo.h>
#include <TinyYolo/app/GraphYolo11Partition.h>
#include <TinyYolo/backend/Esp32NnBackend.h>
#include "yolo26n_graph.inc"  // the table: yolo26::kModel

using namespace tinyyolo;

// The sizes of the export (tools/yolo/yolo_export.py) for the nano model.
const size_t kFloatCount = 2408932;  // float parameters
const size_t kInt8Count = 2397856;   // int8 convolution weights
const size_t kScaleCount = 8516;     // float weight scales
const int kInputSize = 320;  // the network input (the measurements use 320x320)
const bool kPerTensorScales = false;  // true: yolo26::kQuantPerTensor (more accurate, slower)
const bool kP4Kernels = true;         // false: ESP-NN for the convolutions (to compare the two paths)

// The test photo, embedded from photo.rgb by photo.S (raw RGB888).
extern "C" const uint8_t kPhotoData[];
extern "C" const uint8_t kPhotoDataEnd[];
const int kPhotoWidth = 375;
const int kPhotoHeight = 250;

Output out(Serial0);
GraphYolo11Partition weight_blob;
GraphYolo11 detector;
Esp32NnBackend backend;  // int8 convolutions: the P4 assembly kernels (ESP-NN for other shapes)
Detections detections;

void stop() {
  while (true) delay(1000);
}

void setup() {
  Serial0.begin(115200);
  TinyYoloLogger.begin(Serial0, TinyYoloLogLevel::Error);
  delay(8000);  // let the host open the port before the first line
  out.printf("YOLO26n graph on the P4");

  if (kPhotoDataEnd - kPhotoData != kPhotoWidth * kPhotoHeight * 3) {
    out.printf("photo.rgb does not match the photo size");
    stop();
  }
  if (weight_blob.begin("model", kFloatCount, kInt8Count, kScaleCount) != Status::Ok) {
    out.printf("weights not loaded: flash yolo26n_board.bin at 0x310000");
    stop();
  }
  GraphModel model = yolo26::kModel;
  model.input_size = kInputSize;
  if (kPerTensorScales) model.quant = yolo26::kQuantPerTensor;
  if (detector.begin(model, backend, weight_blob.weights(), weight_blob.qweights(), weight_blob.qscales()) != Status::Ok ||
      detections.reserve(100) != Status::Ok) {
    out.printf("detector setup failed");
    stop();
  }
  backend.setP4Kernels(kP4Kernels);
  out.printf("int8 convolutions: %s", backend.p4KernelsEnabled() ? "P4 assembly kernels" : "ESP-NN");
  detector.graph().enableProfiling(true);
}

void loop() {
  static int run_count = 0;
  if (run_count >= 3) {
    delay(1000);
    return;
  }
  ++run_count;

  const ImageView photo(kPhotoData, kPhotoWidth, kPhotoHeight, tinyyolo::PixelFormat::RGB888);
  if (detector.detect(photo, detections) != Status::Ok) {
    out.printf("detect failed");
    return;
  }
  out.printf("run %d: graph %lu ms, head %lu ms, %d detections", run_count,
             (unsigned long)detector.lastGraphMicros() / 1000,
             (unsigned long)detector.lastHeadMicros() / 1000, detections.size());
  for (int i = 0; i < detections.size(); ++i) {
    const Detection& d = detections[i];
    out.printf("  class %d score %.3f box %.1f %.1f %.1f %.1f", d.class_id, d.score, d.x, d.y, d.x + d.width,
               d.y + d.height);
  }
  if (run_count == 1) {
    for (int p = 0; p < (int)Graph::Phase::Count; ++p) {
      const Graph::Phase ph = (Graph::Phase)p;
      out.printf("phase %s: %lu ms", Graph::phaseName(ph), (unsigned long)detector.graph().phaseMicros(ph) / 1000);
    }
  }
  out.printf("free: internal %u, PSRAM %u bytes", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}
