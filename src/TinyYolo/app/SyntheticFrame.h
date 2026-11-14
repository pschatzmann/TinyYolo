#pragma once
/// @file SyntheticFrame.h  Generates test frames without a camera: a gradient with a bright
/// square that moves on every call to next().

#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/image/ImageView.h"

namespace tinyyolo {

class SyntheticFrame {
 public:
  SyntheticFrame(int width, int height) : width_(width), height_(height) {
    if (pixels_.allocate((size_t)width * (size_t)height * 3) != Status::Ok) {
      TINYYOLO_LOGE("SyntheticFrame", "cannot allocate a %dx%d frame", width, height);
    }
  }

  /// Draws the next frame and returns a view of it (RGB888). The view stays valid until the
  /// next call.
  ImageView next() {
    if (pixels_.data() == nullptr) return ImageView();  // the allocation failed (see the constructor)
    const int size = height_ / 4 > 0 ? height_ / 4 : 1;
    const int travel = width_ - size > 0 ? width_ - size : 1;
    const int square = (count_ * 2) % travel;
    const int top = height_ / 3;
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        uint8_t* p = pixels_.data() + ((size_t)y * width_ + x) * 3;
        const bool inside = x >= square && x < square + size && y >= top && y < top + size;
        p[0] = inside ? 255 : (uint8_t)(x * 255 / (width_ > 1 ? width_ - 1 : 1));
        p[1] = inside ? 255 : (uint8_t)(y * 255 / (height_ > 1 ? height_ - 1 : 1));
        p[2] = 128;
      }
    }
    ++count_;
    return ImageView(pixels_.data(), width_, height_, PixelFormat::RGB888);
  }

  /// Number of frames produced so far.
  unsigned long count() const { return count_; }

 private:
  int width_;
  int height_;
  unsigned long count_ = 0;
  Buffer<uint8_t> pixels_;
};

}  // namespace tinyyolo
