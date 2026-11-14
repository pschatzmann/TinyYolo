// Board settings for Yolov4CameraStream. The ESP32-S3-EYE and the Waveshare ESP32-P4-WIFI6 differ
// in the camera pins and in the byte order of the RGB565 frames. Everything chip-specific is here.
#pragma once

#include <TinyCamera.h>
#include <TinyCameraPins.h>
#include <TinyYolo.h>

#if defined(CONFIG_IDF_TARGET_ESP32P4)
constexpr bool kOnP4 = true;
#else
constexpr bool kOnP4 = false;
#endif

/// The S3-EYE camera driver gives RGB565 big-endian; the P4 (ESP_Video) gives little-endian.
constexpr tinyyolo::PixelFormat kCameraFormat =
    kOnP4 ? tinyyolo::PixelFormat::RGB565Little : tinyyolo::PixelFormat::RGB565Big;

/// Sets the camera pins of this board.
inline void setBoardPins(camera_config_t& config) {
#if defined(CONFIG_IDF_TARGET_ESP32P4)
  setPinsWaveshareEsp32P4Wifi6(config);
#else
  setPinsEsp32S3Eye(config);
#endif
}
