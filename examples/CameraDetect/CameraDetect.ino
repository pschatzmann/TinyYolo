// CameraDetect: live object detection on an ESP32-CAM style board. TinyCamera
// captures frames, TinyYolo runs a Darknet model on them.
//
// The model comes from flash (DarknetModel.h, in this folder). The sample
// weights are pseudo-random, so the detections are not meaningful.

#include <TinyCamera.h>
#include <TinyCameraPins.h>
#include <TinyYolo.h>
#include <TinyYolo/TinyCamera/CameraFrame.h>

#include "DarknetModel.h"

Output out(Serial);  // the sketch's text output (Serial)

TinyYoloModel model;
FlashModelSource source(kModelCfg, kModelNames, kModelWeights);
CameraFrame camera_frame(64, 64);  // the network input size

void setup() {
  Serial.begin(115200);
  out.waitForSerialPort(3000);
  TinyYoloLogger.begin(Serial, TinyYoloLogLevel::Info);
  TinyCameraLogger.begin(Serial, TinyCameraLogLevel::Info);

  if (model.begin(source) != Status::Ok) {
    out.printf("Model setup failed; see the log above.");
    while (true) {
      delay(1000);
    }
  }

  camera_config_t config = TinyCamera::defaultConfig();
  config.frame_size = FRAMESIZE_QVGA;
  setPinsAiThinker(config);
  if (!camera_frame.begin(config)) {
    out.printf("Camera init failed");
    while (true) {
      delay(1000);
    }
  }
  out.printf("Camera and model ready");
}

void loop() {
  if (!camera_frame.next()) {
    delay(500);
    return;
  }
  if (model.detect(camera_frame.image()) == Status::Ok) {
    for (const Detection& d : model.detections())
      out.printf("%s %.2f at %d,%d %dx%d",
                     model.detector().label(d.class_id), d.score, (int)d.x,
                     (int)d.y, (int)d.width, (int)d.height);
  }
  delay(500);
}
