// DrawDetections: runs a Darknet model on synthetic frames and shows the frame and its
// detection boxes on an LCD.
//
// Board: LCDBoardGuitionESP32_LVGL_2_4Display (ESP32 Cheap Yellow Display, 240x320). For another
// board, change the board line below.
//
// The model is the built-in flash sample (DarknetModel.h, copied here). Its weights
// are pseudo-random, so the boxes are placed arbitrarily.

#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <TinyGPU.h>
#include <TinyGPU/Boards/LCDBoards.h>
#include <TinyYolo.h>
#include <TinyYolo/TinyGPU/LcdOverlay.h>
#include "DarknetModel.h"

Output out(Serial);  // the sketch's text output (Serial)

LCDBoardGuitionESP32_LVGL_2_4Display board;
LcdOverlay overlay(board);
TinyYoloModel model;
SyntheticFrame frames(64, 64);
FlashModelSource source(kModelCfg, kModelNames, kModelWeights);

void setup() {
  Serial.begin(115200);
  out.waitForSerialPort(3000);
  TinyYoloLogger.begin(Serial, TinyYoloLogLevel::Info);

  overlay.begin();
  if (model.begin(source) != Status::Ok) {
    out.printf("Model setup failed; see the log above.");
    while (true) {
      delay(1000);
    }
  }
}

void loop() {
  ImageView image = frames.next();
  if (model.detect(image) == Status::Ok) {
    overlay.show(image, model.detections());
    for (const Detection& d : model.detections()) out.printf("%s %.2f at %d,%d %dx%d", model.detector().label(d.class_id), d.score, (int)d.x, (int)d.y, (int)d.width, (int)d.height);
  }
  delay(1000);
}
