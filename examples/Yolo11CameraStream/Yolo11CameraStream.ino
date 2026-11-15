// Yolo11CameraStream: YOLO11n on the Waveshare ESP32-P4-WIFI6 camera, streamed to a desktop
// computer. The camera frame is letterboxed to 640x640 and detected with the TinyYolo graph
// (tinyyolo::GraphYolo11, about 6 s per frame); every frame is streamed, with the boxes of the last
// detection. The stream protocol and the viewer are the same as Yolov4CameraStream:
//   tinyyolo --serial /dev/ttyACM0 --rotate 180
//
// The weights are in the "model" partition (partitions.csv, offset 0x310000): flash
// the model file (see README.md).
#include <Arduino.h>
#include <TinyYolo.h>
#include <TinyYolo/TinyCamera/CameraFrame.h>
#include <TinyYolo/app/ModelFilePartition.h>
#include <TinyYolo/backend/Esp32NnBackend.h>
#include <TinyCamera.h>
#include <TinyCameraPins.h>


using namespace tinyyolo;

// The sizes of the export (tools/yolo/yolo_export.py) for the nano model.

const int kCaptureWidth = 320;  // the camera picture; the boxes are in this space
const int kCaptureHeight = 240;
const int kSendWidth = 160;     // the picture sent to the viewer (scaled down)
const int kSendHeight = 120;
const unsigned long kBaud = 921600;
const int kDetectEvery = 1;     // the graph takes about 6 s: every frame is analysed when possible
const tinyyolo::PixelFormat kCameraFormat = tinyyolo::PixelFormat::RGB565Little;  // ESP_Video on the P4

Output out(Serial);  // the text lines of the protocol
ModelFilePartition model_file;  // the model file (docs/ModelFormat.md) in the "model" partition
GraphYolo11 detector;
Esp32NnBackend backend;  // int8 convolutions: the P4 assembly kernels
Detections detections;
CameraFrame camera_frame(kCaptureWidth, kCaptureHeight, kCameraFormat);
uint8_t send_bytes[kSendWidth * kSendHeight * 2];

/// Halts the sketch after a fatal setup error (the error is reported above).
void stop() {
  while (true) delay(1000);
}

/// Starts the camera. Its frame buffers go to PSRAM, and it starts before the model takes memory.
void startCamera() {
  camera_config_t config = TinyCamera::defaultConfig();
  config.frame_size = FRAMESIZE_QVGA;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  setPinsWaveshareEsp32P4Wifi6(config);
  if (!camera_frame.begin(config)) {
    out.printf("camera init failed");
    stop();
  }
}

void setup() {
  Serial.begin(kBaud);
  TinyYoloLogger.begin(Serial, TinyYoloLogLevel::Error);  // errors only
  TinyCameraLogger.begin(Serial, TinyCameraLogLevel::None);

  startCamera();
  if (model_file.begin("model") != Status::Ok) {
    out.printf("weights not loaded: flash the model file at 0x310000");
    stop();
  }
  if (detector.begin(model_file.model(), backend, model_file.floatWeights(), model_file.int8Weights(), model_file.int8Scales()) != Status::Ok ||
      detections.reserve(100) != Status::Ok) {
    out.printf("detector setup failed");
    stop();
  }
}

void loop() {
  static int frame_count = 0;
  if (!camera_frame.next()) return;
  const ImageView picture = camera_frame.image();

  const unsigned long t0 = millis();
  unsigned long detect_ms = 0;
  if (frame_count++ % kDetectEvery == 0) {
    if (detector.detect(picture, detections) != Status::Ok) {
      out.printf("detect failed");
      return;
    }
    detect_ms = millis() - t0;
  }

  // Timing line for diagnostics (the viewer ignores it): detection time and frame time.
  out.printf("T %lu %lu", detect_ms, millis() - t0);
  out.printf("FRAME %d %d %d %d", kSendWidth, kSendHeight, kSendWidth * kSendHeight * 2, detections.size());
  for (const Detection& d : detections) {
    out.printf("%.1f %.1f %.1f %.1f %.3f %d", d.x, d.y, d.width, d.height, d.score, d.class_id);
  }
  // The picture for the viewer: the camera frame scaled to the stream size, little-endian.
  const ImageView small = picture.scaleTo(send_bytes, kSendWidth, kSendHeight, tinyyolo::PixelFormat::RGB565Little);
  Serial.write(small.data(), small.byteCount());
  Serial.flush();
}
