#pragma once
/// @file CameraFrame.h  Frames from a TinyCamera, scaled to the network input (RGB565).
///
/// Requires the TinyCamera library. Include this header after <TinyCamera.h>.

#if !__has_include(<TinyCamera.h>)
#error "TinyYolo/TinyCamera/CameraFrame.h needs the TinyCamera library: install it (https://github.com/pschatzmann/TinyCamera) or remove this include."
#endif

#include <TinyCamera.h>
#include <TinyCameraConvert.h>
#include <TinyCameraPins.h>

// On Arduino, sketches use the camera names without the tiny_camera:: prefix. Only these names
// are imported (not the whole namespace), so tinyyolo::PixelFormat and tiny_camera::PixelFormat
// do not clash.
#if defined(ARDUINO)
using tiny_camera::TinyCamera;
using tiny_camera::TinyCameraLogLevel;
using tiny_camera::TinyCameraLogger;
using tiny_camera::setPinsAiThinker;
using tiny_camera::setPinsEsp32S3Eye;
#if defined(TINY_CAMERA_ESP_VIDEO)  // the P4 pin preset exists only with the ESP_Video driver
using tiny_camera::setPinsWaveshareEsp32P4Wifi6;
#endif
#endif

#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/image/ImageView.h"

namespace tinyyolo {

class CameraFrame {
 public:
  /// width x height is the network input size. `format` is the byte order of the camera's
  /// RGB565 output (see PixelFormat).
  CameraFrame(int width, int height, PixelFormat format = PixelFormat::RGB565Big)
      : width_(width), height_(height), format_(format) {
    if (buffer_.allocate((size_t)width * (size_t)height * 2) != Status::Ok) {
      TINYYOLO_LOGE("CameraFrame", "cannot allocate a %dx%d frame", width, height);
    }
  }

  /// Starts the camera. `config` comes from TinyCamera::defaultConfig() plus the board's pins.
  /// The pixel format is set to RGB565 here.
  bool begin(camera_config_t config) {
    config.pixel_format = PIXFORMAT_RGB565;
    return camera_.begin(config);
  }

  /// Captures and scales the next frame. Returns false on failure (logged).
  bool next() {
    tiny_camera::TinyCameraFrame frame = camera_.captureFrame();
    if (!frame) {
      TINYYOLO_LOGW("CameraFrame", "capture failed");
      return false;
    }
    if (!tiny_camera::scaleRgb565(frame, buffer_.data(), (size_t)width_, (size_t)height_)) {
      TINYYOLO_LOGW("CameraFrame", "scaling the frame failed");
      return false;
    }
    return true;  // the camera buffer is returned when `frame` goes out of scope
  }

  /// View of the last frame returned by next().
  ImageView image() const { return ImageView(buffer_.data(), width_, height_, format_); }

 private:
  tiny_camera::TinyCamera camera_;
  Buffer<uint8_t> buffer_;
  int width_;
  int height_;
  PixelFormat format_;
};

}  // namespace tinyyolo
