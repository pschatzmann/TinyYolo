// CameraStream: runs YOLO11n (int8, graph path) on a camera board and streams the picture
// and the detections to a desktop computer over the USB-UART link (no display needed).
// YOLO11n is the fastest model on the ESP32-S3 (about 6.6 s per frame).
//
// Board support: everything that depends on the board is in the "board" section below.
// The model is a single file (docs/ModelFormat.md) in the "model" flash partition (partitions.csv,
// offset 0x310000). It is written by the export (tools/yolo/yolo_export.py)
// and flash it there (see README.md). The example needs an ESP32 with a 16 MB flash and PSRAM.
// The camera pins are those of the ESP32-S3-EYE.
// The desktop tool reads the stream: tinyyolo --serial /dev/ttyACM0
//
// Frame protocol (binary-safe, 921600 baud, 8N1):
//   "MODEL <name>\n"                          the model name, before every frame
//   "JPEG <w> <h> <nbytes> <ndet>\n"         header: the picture is the camera's JPEG, w x h
//   ndet lines "x y width height score class\n"   boxes in the w x h picture
//   nbytes bytes: the JPEG picture, as the camera delivered it
//
// The camera delivers JPEG frames (an RGB565 DMA capture does not initialize on the
// S3-EYE); each frame is decoded to RGB565 for the detector, and the JPEG is sent as it is.
//
// Logging is off, because log lines would corrupt the stream.

#include <TinyCamera.h>
#include <TinyCameraConvert.h>
#include <TinyCameraPins.h>
#include <TinyYolo.h>

// ---- board ----------------------------------------------------------------------------
#if defined(ESP32)
#include <TinyYolo/app/ModelFilePartition.h>
#include <TinyYolo/backend/Esp32NnBackend.h>
using WeightsSource = tinyyolo::ModelFilePartition;  // the "model" partition, a model file
using BoardBackend = tinyyolo::Esp32NnBackend;       // the S3 int8 kernels (ESP-NN)
#else
#error "CameraStream: this board has no camera and weight source set up. Add them in the board section."
#endif

/// Starts the camera, JPEG frames of QVGA size, the frame buffers in PSRAM.
bool startCamera(TinyCamera& camera) {
#if defined(ESP32)
  camera_config_t config = TinyCamera::defaultConfig();  // JPEG output
  config.frame_size = FRAMESIZE_QVGA;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  tiny_camera::setPinsEsp32S3Eye(config);
  return camera.begin(config);
#endif
}
// ---------------------------------------------------------------------------------------

const char* kModelName = "YOLO11n int8";
const int kInputSize = 320;  // the network input (the measurements use 320x320)
const int kCaptureWidth = 320;
const int kCaptureHeight = 240;
const unsigned long kBaud = 921600;

using namespace tinyyolo;

Output out(Serial);  // the sketch's text output (Serial)
WeightsSource model_file;  // the graph and the weights, from the "model" partition
GraphYolo11 detector;
BoardBackend backend;
TinyCamera camera;
Buffer<uint8_t> rgb565;  // the decoded picture (PSRAM when it is large)
Detections detections;

/// Halts the sketch after a fatal setup error.
void stop() {
  while (true) delay(1000);
}

void setup() {
  Serial.begin(kBaud);
  TinyCameraLogger.begin(Serial, TinyCameraLogLevel::None);

  if (model_file.begin("model") != Status::Ok) {
    stop();  // flash the model file at 0x310000
  }
  GraphModel model = model_file.model();
  model.input_size = kInputSize;
  if (detector.begin(model, backend, model_file.floatWeights(), model_file.int8Weights(),
                     model_file.int8Scales()) != Status::Ok ||
      detections.reserve(100) != Status::Ok) {
    stop();
  }
  if (rgb565.allocate(kCaptureWidth * kCaptureHeight * 2, MemoryPlacement::PsramFirst) != Status::Ok) {
    stop();
  }
  if (!startCamera(camera)) {
    stop();
  }
}

void loop() {
  TinyCameraFrame frame = camera.captureFrame();
  if (!frame) return;
  if (!toRgb565(frame, rgb565.data())) return;
  const ImageView picture(rgb565.data(), kCaptureWidth, kCaptureHeight, tinyyolo::PixelFormat::RGB565Little);
  if (detector.detect(picture, detections) != Status::Ok) return;

  out.printf("MODEL %s", kModelName);
  out.printf("JPEG %d %d %d %d", kCaptureWidth, kCaptureHeight, (int)frame.size(), detections.size());
  for (int i = 0; i < detections.size(); ++i) {
    const Detection& d = detections[i];
    out.printf("%.1f %.1f %.1f %.1f %.3f %d", d.x, d.y, d.width, d.height, d.score, d.class_id);
  }
  // The picture for the viewer: the camera's JPEG as it came from the camera (the desktop decodes it).
  Serial.write(frame.data(), frame.size());
  Serial.flush();
}
