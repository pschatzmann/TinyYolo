#pragma once
/// @file ImageView.h  Non-owning view of a camera or frame-buffer image.

#include <stdint.h>

#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"

namespace tinyyolo {

enum class PixelFormat : uint8_t {
  Gray8,        ///< 1 byte per pixel
  RGB888,       ///< 3 bytes: R G B
  BGR888,       ///< 3 bytes: B G R
  RGB565Big,    ///< 2 bytes, high byte first (typical camera output)
  RGB565Little  ///< 2 bytes, low byte first
};

class ImageView {
 public:
  ImageView() {}

  /// row_bytes = 0 means rows are tightly packed.
  ImageView(const uint8_t* data, int width, int height, PixelFormat format, int row_bytes = 0)
      : data_(data),
        width_(width),
        height_(height),
        format_(format),
        row_bytes_(row_bytes > 0 ? row_bytes : width * bytesPerPixel(format)) {}

  static int bytesPerPixel(PixelFormat format) {
    switch (format) {
      case PixelFormat::Gray8: return 1;
      case PixelFormat::RGB565Big:
      case PixelFormat::RGB565Little: return 2;
      default: return 3;
    }
  }

  /// Logs an error and returns a failure status if the view cannot be sampled.
  Status validate() const {
    if (data_ == nullptr) {
      TINYYOLO_LOGE("Image", "image data is null");
      return Status::InvalidArgument;
    }
    if (width_ <= 0 || height_ <= 0) {
      TINYYOLO_LOGE("Image", "invalid image size %dx%d", width_, height_);
      return Status::InvalidArgument;
    }
    if (row_bytes_ < width_ * bytesPerPixel(format_)) {
      TINYYOLO_LOGE("Image", "row stride %d is smaller than one row (%d bytes)", row_bytes_,
                    width_ * bytesPerPixel(format_));
      return Status::InvalidArgument;
    }
    return Status::Ok;
  }

  int width() const { return width_; }
  int height() const { return height_; }
  PixelFormat format() const { return format_; }

  /// RGB of pixel (x, y) with components in [0, 1]. Coordinates must lie inside the image.
  void rgbAt(int x, int y, float rgb[3]) const {
    const uint8_t* p = data_ + (size_t)y * (size_t)row_bytes_ + (size_t)x * bytesPerPixel(format_);
    switch (format_) {
      case PixelFormat::Gray8:
        rgb[0] = rgb[1] = rgb[2] = p[0] / 255.0f;
        break;
      case PixelFormat::RGB888:
        rgb[0] = p[0] / 255.0f;
        rgb[1] = p[1] / 255.0f;
        rgb[2] = p[2] / 255.0f;
        break;
      case PixelFormat::BGR888:
        rgb[0] = p[2] / 255.0f;
        rgb[1] = p[1] / 255.0f;
        rgb[2] = p[0] / 255.0f;
        break;
      case PixelFormat::RGB565Big:
        unpack565((uint16_t)((p[0] << 8) | p[1]), rgb);
        break;
      case PixelFormat::RGB565Little:
        unpack565((uint16_t)((p[1] << 8) | p[0]), rgb);
        break;
    }
  }

  /// Nearest-neighbour scaling into `dst`: a packed buffer of width x height RGB565 pixels in
  /// byte order `format` (RGB565Big or RGB565Little). Returns an ImageView of that buffer, so
  /// the result can be used directly (for example data() and byteCount() for Serial.write()).
  /// An RGB565 source is copied as 16-bit values, so no float conversion is needed; other sources
  /// are converted through rgbAt(). `dst` must hold width * height * 2 bytes. On error the
  /// returned view is empty (data() is nullptr).
  ImageView scaleTo(uint8_t* dst, int width, int height, PixelFormat format) const {
    if (dst == nullptr || width <= 0 || height <= 0 || !isRgb565(format)) {
      TINYYOLO_LOGE("ImageView", "scaleTo: needs a destination and an RGB565 format");
      return ImageView();
    }
    if (validate() != Status::Ok) return ImageView();
    const bool big_out = format == PixelFormat::RGB565Big;
    for (int y = 0; y < height; ++y) {
      // Sample at the centre of each destination pixel.
      const int sy = (int)(((2 * (int64_t)y + 1) * height_) / (2 * (int64_t)height));
      for (int x = 0; x < width; ++x) {
        const int sx = (int)(((2 * (int64_t)x + 1) * width_) / (2 * (int64_t)width));
        const uint16_t v = rawRgb565(sx, sy);
        uint8_t* d = dst + ((size_t)y * width + x) * 2;
        d[big_out ? 0 : 1] = (uint8_t)(v >> 8);
        d[big_out ? 1 : 0] = (uint8_t)(v & 0xFF);
      }
    }
    return ImageView(dst, width, height, format, width * 2);
  }

  /// Pixel data (nullptr for an empty view) and its size in bytes (rows as stored, no padding
  /// after the last row).
  const uint8_t* data() const { return data_; }
  size_t byteCount() const { return data_ == nullptr ? 0 : (size_t)row_bytes_ * (size_t)height_; }

  static bool isRgb565(PixelFormat format) {
    return format == PixelFormat::RGB565Big || format == PixelFormat::RGB565Little;
  }

 private:
  /// The pixel as a native 16-bit RGB565 value (converted from other formats when needed).
  uint16_t rawRgb565(int x, int y) const {
    const uint8_t* p = data_ + (size_t)y * (size_t)row_bytes_ + (size_t)x * bytesPerPixel(format_);
    if (format_ == PixelFormat::RGB565Big) return (uint16_t)((p[0] << 8) | p[1]);
    if (format_ == PixelFormat::RGB565Little) return (uint16_t)((p[1] << 8) | p[0]);
    float rgb[3];
    rgbAt(x, y, rgb);
    return (uint16_t)(((int)(rgb[0] * 31.0f + 0.5f) << 11) | ((int)(rgb[1] * 63.0f + 0.5f) << 5) |
                      (int)(rgb[2] * 31.0f + 0.5f));
  }

  static void unpack565(uint16_t v, float rgb[3]) {
    rgb[0] = ((v >> 11) & 0x1F) / 31.0f;
    rgb[1] = ((v >> 5) & 0x3F) / 63.0f;
    rgb[2] = (v & 0x1F) / 31.0f;
  }

  const uint8_t* data_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  PixelFormat format_ = PixelFormat::RGB888;
  int row_bytes_ = 0;
};

}  // namespace tinyyolo
