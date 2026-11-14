// CameraStream: runs the detector on an ESP32-S3 camera board and streams the
// picture and the detections to a desktop computer over the USB-UART link (no
// display needed).
//
// Board: ESP32-S3-EYE pin preset (the board used in TinyCamera's examples),
// PSRAM=opi. The desktop viewer reads the stream: camera_view --serial
// /dev/ttyACM0
//
// Frame protocol (binary-safe, 921600 baud, 8N1):
//   "FRAME <w> <h> <nbytes> <ndet>\n"        header, picture size 160x120 here
//   ndet lines "x y width height score class\n"   boxes in 320x240 picture
//   coordinates nbytes bytes: the picture as RGB565 little-endian, row by row
//
// Logging is off, because log lines would corrupt the stream.

#include <TinyCamera.h>
#include <TinyCameraPins.h>
#include <TinyYolo.h>
#include <TinyYolo/TinyCamera/CameraFrame.h>

#include "DarknetModel.h"

Output out(Serial);  // the sketch's text output (Serial)
const int kCaptureWidth = 320;
const int kCaptureHeight = 240;
const int kSendWidth = 160;  // every second pixel is sent
const int kSendHeight = 120;
const unsigned long kBaud = 921600;

TinyYoloModel model;
FlashModelSource source(kModelCfg, kModelNames, kModelWeights);
CameraFrame camera_frame(kCaptureWidth, kCaptureHeight,
                         tinyyolo::PixelFormat::RGB565Big);
Detections detections;
uint8_t send_bytes[kSendWidth * kSendHeight * 2];

/// Halts the sketch after a fatal setup error (the error is reported above).
void stop() {
  while (true) delay(1000);
}

void setup() {
  Serial.begin(kBaud);
  TinyCameraLogger.begin(Serial, TinyCameraLogLevel::None);

  bool ok = model.begin(source) == Status::Ok;
  if (!ok) {
    stop();
  }
  detections.reserve(20);

  camera_config_t config = TinyCamera::defaultConfig();
  config.frame_size = FRAMESIZE_QVGA;
  setPinsEsp32S3Eye(config);
  if (!camera_frame.begin(config)) {
    stop();
  }
}

void loop() {
  if (!camera_frame.next()) return;
  const ImageView picture = camera_frame.image();
  if (model.detect(picture) != Status::Ok) return;
  const Detections& found = model.detections();

  out.printf("FRAME %d %d %d %d", kSendWidth, kSendHeight,
             kSendWidth * kSendHeight * 2, found.size());
  for (const Detection& d : found) {
    out.printf("%.1f %.1f %.1f %.1f %.3f %d", d.x, d.y, d.width, d.height,
               d.score, d.class_id);
  }
  // The picture for the viewer: the camera frame scaled to the stream size,
  // little-endian.
  const ImageView small = picture.scaleTo(send_bytes, kSendWidth, kSendHeight,
                                          tinyyolo::PixelFormat::RGB565Little);
  Serial.write(small.data(), small.byteCount());
  Serial.flush();
}
