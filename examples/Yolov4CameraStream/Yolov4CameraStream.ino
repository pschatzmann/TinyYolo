// Yolov4CameraStream: YOLOv4-tiny on a camera board, streamed to a desktop computer.
// Boards: ESP32-S3-EYE (PSRAM=opi, FlashSize=8M, PartitionScheme=custom) and
// Waveshare ESP32-P4-WIFI6 (FlashSize=32M, ChipVariant=postv3, PartitionScheme=custom).
//
// The model is int8 with 192x192 input. Its weights stay in the flash partition "model" and are
// read in place (on the P4 they are copied into PSRAM first). Flash the weights once, after the
// sketch, at offset 0x140000 (the "model" partition in partitions.csv):
//   esptool.py --chip esp32s3 --port /dev/ttyACM0 write_flash 0x140000 yolov4-tiny.blob
// (use --chip esp32p4 on the P4; yolov4-tiny.blob is made with tinyyolo_export_blob.)
//
// The stream goes to the desktop viewer: camera_view --serial /dev/ttyACM0 --rotate 180
// Protocol: see CameraStream.ino.

#include <TinyYolo.h>
#include <TinyYolo/TinyCamera/CameraFrame.h>
#include <TinyYolo/app/PartitionModel.h>
#include <TinyYolo/backend/Esp32NnBackend.h>
#include "Board.h"
#include "Yolov4Model.h"

const int kCaptureWidth = 320;
const int kCaptureHeight = 240;
const int kSendWidth = 160;
const int kSendHeight = 120;
const unsigned long kBaud = 921600;
const int kDetectEvery = 3;  // the detector runs on every third camera frame

Output out(Serial);  // the sketch's text output (Serial)
Detector detector;
Esp32NnBackend backend;  // ESP-NN int8 convolutions
PartitionModel model_store;  // keeps the mapped "model" partition alive
Detections detections;
CameraFrame camera_frame(kCaptureWidth, kCaptureHeight, kCameraFormat);
uint8_t send_bytes[kSendWidth * kSendHeight * 2];

/// Builds the detector from the "model" partition. Errors are logged by the library.
bool loadModel() {
  DetectorConfig config;
  config.weight_precision = WeightPrecision::Int8;
  config.score_threshold = 0.3f;
  config.max_candidates = 512;
  return model_store.begin(detector, kModelCfg, sizeof(kModelCfg), kModelNames,
                           sizeof(kModelNames), config, &backend, "model", kOnP4) == Status::Ok;
}

/// Halts the sketch after a fatal setup error (the error is reported above).
void stop() {
  while (true) delay(1000);
}

/// Starts the camera. Its frame buffers go to PSRAM, and it starts before the model takes memory.
void startCamera() {
  camera_config_t config = TinyCamera::defaultConfig();
  config.frame_size = FRAMESIZE_QVGA;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  setBoardPins(config);
  if (!camera_frame.begin(config)) {
    out.printf("camera init failed");
    stop();
  }
}

/// Int8 activation scales from the first camera frame (see Detector::calibrate).
void calibrateModel() {
  while (!camera_frame.next()) delay(10);
  const unsigned long start = millis();
  if (detector.calibrate(camera_frame.image()) != Status::Ok) {
    out.printf("calibration failed");
    stop();
  }
  out.printf("calibrated in %lu ms", millis() - start);
}

void reportMemory() {
  out.printf("free after setup: internal %u bytes, PSRAM %u bytes",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

void setup() {
  Serial.begin(kBaud);
  TinyYoloLogger.begin(Serial, TinyYoloLogLevel::Error);  // errors only; no output when all is well
  TinyCameraLogger.begin(Serial, TinyCameraLogLevel::None);

  startCamera();
  if (!loadModel()) stop();
  calibrateModel();
  reportMemory();
  detections.reserve(40);
}

void loop() {
  static int frame_count = 0;
  if (!camera_frame.next()) return;
  const ImageView picture = camera_frame.image();

  // Detection is slow, so only every kDetectEvery-th frame is analysed. Every frame is streamed,
  // with the boxes of the last detection.
  const unsigned long t0 = millis();
  unsigned long detect_ms = 0;
  if (frame_count++ % kDetectEvery == 0) {
    detector.detect(picture, detections);
    detect_ms = millis() - t0;
  }

  // Timing line for diagnostics (the viewer ignores it): detection time and frame time.
  out.printf("T %lu %lu", detect_ms, millis() - t0);
  out.printf("FRAME %d %d %d %d", kSendWidth, kSendHeight, kSendWidth * kSendHeight * 2,
             detections.size());
  for (const Detection& d : detections) {
    out.printf("%.1f %.1f %.1f %.1f %.3f %d", d.x, d.y, d.width, d.height, d.score, d.class_id);
  }
  // The picture for the viewer: the camera frame scaled to the stream size, little-endian.
  const ImageView small =
      picture.scaleTo(send_bytes, kSendWidth, kSendHeight, tinyyolo::PixelFormat::RGB565Little);
  Serial.write(small.data(), small.byteCount());
  Serial.flush();
}
