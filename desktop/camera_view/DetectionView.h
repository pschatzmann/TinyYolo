#pragma once
/// @file DetectionView.h  A TinyMaterialDesign widget that shows a camera picture with its
/// detection boxes and a list of the results.
///
/// Requires the TinyMaterialDesign library (and TinyGPU). Include this header after
/// <TinyMaterialDesign.h>.

#if !__has_include(<TinyMaterialDesign.h>)
#error "DetectionView.h needs the TinyMaterialDesign library: install it (https://github.com/pschatzmann/TinyMaterialDesign) or remove this include."
#endif

#include <TinyMaterialDesign.h>
#include <stdio.h>

#include "TinyYolo/app/TinyYoloModel.h"
#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/image/ImageView.h"

namespace tinyyolo {

template <typename RGB_T>
class DetectionView : public tinymd::Widget<RGB_T> {
 public:
  static constexpr int kMaxBoxes = 20;

  /// The picture is picture_w x picture_h pixels, drawn at the top-left of `bounds`. The
  /// result list is drawn below it. `model` provides the class labels.
  DetectionView(const tinymd::Bounds& bounds, int picture_w, int picture_h, const TinyYoloModel& model)
      : picture_w_(picture_w), picture_h_(picture_h), model_(model) {
    this->bounds = bounds;
    if (pixels_.allocate((size_t)picture_w * (size_t)picture_h * 3) != Status::Ok) {
      TINYYOLO_LOGE("DetectionView", "cannot allocate the picture buffer");
    }
  }

  /// Copies a new picture and its detections, and asks for a redraw. The picture is copied,
  /// so the camera buffer can be reused right away.
  void setFrame(const ImageView& picture, const Detections& detections) {
    setPicture(picture);
    setDetections(detections);
  }

  /// Copies a new picture; the boxes stay as they are.
  void setPicture(const ImageView& picture) {
    if (picture.width() == picture_w_ && picture.height() == picture_h_ && pixels_.data() != nullptr) {
      uint8_t* p = pixels_.data();
      for (int y = 0; y < picture_h_; ++y) {
        for (int x = 0; x < picture_w_; ++x) {
          float rgb[3];
          picture.rgbAt(x, y, rgb);
          p[0] = (uint8_t)(rgb[0] * 255.0f + 0.5f);
          p[1] = (uint8_t)(rgb[1] * 255.0f + 0.5f);
          p[2] = (uint8_t)(rgb[2] * 255.0f + 0.5f);
          p += 3;
        }
      }
    }
    changed_ = true;
  }

  /// Replaces the boxes and the result list; the picture stays as it is.
  void setDetections(const Detections& detections) {
    setDetections(detections.begin(), detections.size());
  }

  /// Same, from an array of `count` detections.
  void setDetections(const Detection* items, int count) {
    count_ = 0;
    for (int i = 0; i < count && count_ < kMaxBoxes; ++i) items_[count_++] = items[i];
    changed_ = true;
  }

  // --- tinymd::Widget ----------------------------------------------------------

  bool update(uint32_t) override { return changed_; }

  void draw(tinygpu::ISurface<RGB_T>& target) override {
    changed_ = false;
    const int32_t x0 = this->bounds.x;
    const int32_t y0 = this->bounds.y;
    target.fillRect((size_t)x0, (size_t)y0, (size_t)this->bounds.w, (size_t)this->bounds.h,
                    RGB_T(30, 30, 30));

    // Picture.
    if (pixels_.data() != nullptr) {
      const uint8_t* p = pixels_.data();
      for (int y = 0; y < picture_h_; ++y) {
        for (int x = 0; x < picture_w_; ++x) {
          target.setPixel((size_t)(x0 + x), (size_t)(y0 + y), RGB_T(p[0], p[1], p[2]));
          p += 3;
        }
      }
    }

    // Boxes and their labels.
    const RGB_T red(255, 0, 0);
    const RGB_T white(255, 255, 255);
    const RGB_T black(0, 0, 0);
    char text[64];
    for (int i = 0; i < count_; ++i) {
      const Detection& d = items_[i];
      target.drawRect((size_t)(x0 + (int)d.x), (size_t)(y0 + (int)d.y), (size_t)d.width,
                      (size_t)d.height, red);
      snprintf(text, sizeof(text), "%s %.2f", model_.label(d.class_id), d.score);
      const int32_t ty = (int32_t)(y0 + (int)d.y) - 10;
      target.drawText((int16_t)(x0 + (int)d.x), (int16_t)(ty < y0 ? y0 : ty), text, white, black, true);
    }

    // Result list below the picture.
    const int32_t list_top = y0 + picture_h_ + 6;
    const int lines = (int)((this->bounds.y + this->bounds.h - list_top) / 12);
    if (count_ == 0) {
      target.drawText((int16_t)x0 + 6, (int16_t)list_top, "no detections", white, black, false);
    }
    for (int i = 0; i < count_ && i < lines; ++i) {
      const Detection& d = items_[i];
      snprintf(text, sizeof(text), "%s %.2f  at %d,%d", model_.label(d.class_id), d.score,
               (int)d.x, (int)d.y);
      target.drawText((int16_t)(x0 + 6), (int16_t)(list_top + 12 * i), text, white, black, false);
    }
  }

 private:
  int picture_w_;
  int picture_h_;
  const TinyYoloModel& model_;
  Buffer<uint8_t> pixels_;
  Detection items_[kMaxBoxes];
  int count_ = 0;
  bool changed_ = true;
};

}  // namespace tinyyolo
