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
#include <string.h>

#include <functional>
#include <type_traits>


#include "TinyYolo.h"
#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/image/ImageView.h"

namespace tinyyolo {

template <typename RGB_T>
class DetectionView : public tinymd::Widget<RGB_T> {
 public:
  static constexpr int kMaxBoxes = 20;

  /// Returns the name of a class (any model: Darknet or a YOLO graph).
  using LabelFunction = std::function<const char*(int class_id)>;

  /// The picture is picture_w x picture_h pixels, drawn at the top-left of `bounds`. The
  /// result list is drawn below it. `label` gives the class names.
  DetectionView(const tinymd::Bounds& bounds, int picture_w, int picture_h, LabelFunction label)
      : picture_w_(picture_w), picture_h_(picture_h), label_(std::move(label)) {
    this->bounds = bounds;
    if (pixels_.allocate((size_t)picture_w * (size_t)picture_h) != Status::Ok) {
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
      RGB_T* out = pixels_.data();
      const bool raw565 = picture.format() == PixelFormat::RGB565Little;
      for (int y = 0; y < picture_h_; ++y) {
        if (raw565) {
          // The camera's own RGB565 bytes: no conversion, one 16-bit pixel at a time.
          const uint8_t* src = picture.data() + (size_t)y * (size_t)picture.rowBytes();
          for (int x = 0; x < picture_w_; ++x, src += 2) *out++ = pixelFromRgb565Little(src);
        } else {
          for (int x = 0; x < picture_w_; ++x) {
            float rgb[3];
            picture.rgbAt(x, y, rgb);
            *out++ = RGB_T((uint8_t)(rgb[0] * 255.0f + 0.5f), (uint8_t)(rgb[1] * 255.0f + 0.5f),
                           (uint8_t)(rgb[2] * 255.0f + 0.5f));
          }
        }
      }
    }
    changed_ = true;
  }

  /// The model, its settings (detail) and the picture source, shown in cards below the picture
  /// (optional).
  void setInfo(const char* model, const char* detail, const char* source) {
    snprintf(model_, sizeof(model_), "%s", model ? model : "");
    snprintf(detail_, sizeof(detail_), "%s", detail ? detail : "");
    snprintf(source_, sizeof(source_), "%s", source ? source : "");
    has_info_ = true;
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
      // Whole rows are copied into the target's own buffer when it is a RAM surface of this pixel
      // type (the stored pixels are the same bytes); other surfaces get setPixel().
      const size_t bpp = RGB_T::size() / 8;
      const size_t tw = target.width(), th = target.height();
      uint8_t* base = const_cast<uint8_t*>(target.data());  // the target's buffer, row-major
      const bool direct = base != nullptr && sizeof(RGB_T) == bpp && (size_t)target.size() >= tw * th * bpp &&
                          (size_t)(x0 + picture_w_) <= tw && (size_t)(y0 + picture_h_) <= th;
      if (direct) {
        for (int y = 0; y < picture_h_; ++y) {
          memcpy(base + (((size_t)(y0 + y) * tw) + (size_t)x0) * bpp,
                 pixels_.data() + (size_t)y * picture_w_, (size_t)picture_w_ * bpp);
        }
      } else {
        const RGB_T* p = pixels_.data();
        for (int y = 0; y < picture_h_; ++y) {
          for (int x = 0; x < picture_w_; ++x) target.setPixel((size_t)(x0 + x), (size_t)(y0 + y), *p++);
        }
      }
    }

    // Boxes and their labels, in the colour of the class.
    const RGB_T white(255, 255, 255);
    const RGB_T black(0, 0, 0);
    char text[64];
    for (int i = 0; i < count_; ++i) {
      const Detection& d = items_[i];
      // The box is clipped to the picture: a box that reaches over the edge would wrap around as an
      // unsigned size and draw forever.
      const int32_t bx1 = (int32_t)d.x < 0 ? 0 : (int32_t)d.x;
      const int32_t by1 = (int32_t)d.y < 0 ? 0 : (int32_t)d.y;
      const int32_t bx2 = (int32_t)(d.x + d.width) > picture_w_ ? picture_w_ : (int32_t)(d.x + d.width);
      const int32_t by2 = (int32_t)(d.y + d.height) > picture_h_ ? picture_h_ : (int32_t)(d.y + d.height);
      if (!(d.width > 0 && d.height > 0) || bx2 <= bx1 || by2 <= by1) continue;  // also NaN boxes
      target.drawRect((size_t)(x0 + bx1), (size_t)(y0 + by1), (size_t)(bx2 - bx1), (size_t)(by2 - by1),
                      swatch(d.class_id));
      snprintf(text, sizeof(text), "%s %.2f", label_(d.class_id), d.score);
      const int32_t ty = (int32_t)(y0 + (int)d.y) - 10;
      target.drawText((int16_t)(x0 + (int)d.x), (int16_t)(ty < y0 ? y0 : ty), text, white, black, true);
    }

    // The cards below the picture.
    const int32_t card_x = x0 + 4;
    const int32_t card_y = y0 + picture_h_ + 4;
    const int32_t card_w = this->bounds.w - 8;
    const int32_t card_bottom = this->bounds.y + this->bounds.h - 4;
    const int32_t card_h = card_bottom - card_y;
    if (card_h < 40) return;

    static tinygpu::Font12x12<RGB_T> title_font;
    static tinygpu::Font8x12<RGB_T> body_font;
    static tinygpu::Font16x24<RGB_T> number_font;
    const int32_t pad = 10;
    const int32_t card_w_inner = card_w - 2 * pad;
    const int32_t line_h = (int32_t)body_font.getHeight(1);
    const int32_t title_h = (int32_t)title_font.getHeight(1);

    // One row per detection: swatch, class, score bar and percentage.
    auto drawRows = [&](int32_t top, int32_t bottom) {
      const int32_t row_h = 18;
      const int32_t rows = (bottom - top) / row_h;
      for (int i = 0; i < count_ && i < rows; ++i) {
        const Detection& d = items_[i];
        const int32_t ry = top + i * row_h;
        const RGB_T color = swatch(d.class_id);
        target.fillRoundRect((size_t)(card_x + pad), (size_t)(ry + 3), 10, 10, 3, color);
        char name[24];
        snprintf(name, sizeof(name), "%.14s", label_(d.class_id));
        body_font.drawText(target, (int16_t)(card_x + pad + 16), (int16_t)ry, name, kOnCard, kCard, false);
        const int32_t bar_w = 72;
        const int32_t bar_x = card_x + card_w - pad - 40 - bar_w;
        const float score = d.score < 0 ? 0 : (d.score > 1 ? 1 : d.score);
        target.fillRoundRect((size_t)bar_x, (size_t)(ry + 6), (size_t)bar_w, 5, 2, kTrack);
        const int32_t fill = (int32_t)(bar_w * score + 0.5f);
        if (fill > 0) target.fillRoundRect((size_t)bar_x, (size_t)(ry + 6), (size_t)fill, 5, 2, color);
        char pct[8];
        snprintf(pct, sizeof(pct), "%3d%%", (int)(score * 100 + 0.5f));
        body_font.drawText(target, (int16_t)(card_x + card_w - pad - 34), (int16_t)ry, pct, kOnCard, kCard, false);
      }
      if (count_ > rows && rows > 0) {
        snprintf(text, sizeof(text), "+ %d more", count_ - rows);
        body_font.drawText(target, (int16_t)(card_x + pad), (int16_t)(bottom - line_h), text, kCardMuted, kCard, false);
      }
      if (count_ == 0) {
        body_font.drawText(target, (int16_t)(card_x + pad), (int16_t)(top + 2), "No detections", kCardMuted, kCard, false);
      }
    };
    auto card = [&](int32_t x, int32_t y, int32_t w, int32_t h) {
      target.fillRoundRect((size_t)x, (size_t)y, (size_t)w, (size_t)h, 10, kCard);
      target.drawRoundRect((size_t)x, (size_t)y, (size_t)w, (size_t)h, 10, kTrack);
    };

    if (has_info_) {
      // Three cards: the model (left, top) and the count (right, top), then the results.
      const int32_t top_h = 50;
      const int32_t gap = 4;
      const int32_t info_w = card_w * 2 / 3;
      const int32_t count_w = card_w - info_w - gap;
      card(card_x, card_y, info_w, top_h);
      int32_t y = card_y + 6;
      title_font.drawText(target, (int16_t)(card_x + pad), (int16_t)y, model_, kOnCard, kCard, false);
      y += title_h + 2;
      body_font.drawText(target, (int16_t)(card_x + pad), (int16_t)y, detail_, kOnCard, kCard, false);
      y += line_h;
      snprintf(text, sizeof(text), "%.30s", source_);
      body_font.drawText(target, (int16_t)(card_x + pad), (int16_t)y, text, kCardMuted, kCard, false);

      const int32_t count_x = card_x + info_w + gap;
      card(count_x, card_y, count_w, top_h);
      snprintf(text, sizeof(text), "%d", count_);
      const int32_t number_w = 16 * (int32_t)strlen(text);  // Font16x24: 16 pixels per character
      number_font.drawText(target, (int16_t)(count_x + (count_w - number_w) / 2), (int16_t)(card_y + 4), text,
                           kOnCard, kCard, false);
      const char* caption = count_ == 1 ? "detection" : "detections";
      const int32_t caption_w = 8 * (int32_t)strlen(caption);  // Font8x12: 8 pixels per character
      body_font.drawText(target, (int16_t)(count_x + (count_w - caption_w) / 2), (int16_t)(card_y + 32), caption,
                         kCardMuted, kCard, false);

      const int32_t results_top = card_y + top_h + gap;
      card(card_x, results_top, card_w, card_bottom - results_top);
      drawRows(results_top + 6, card_bottom - 4);
    } else {
      // One card: the count as its title, then the rows.
      card(card_x, card_y, card_w, card_h);
      int32_t y = card_y + 6;
      snprintf(text, sizeof(text), count_ == 0 ? "No detections" : "%d detections", count_);
      title_font.drawText(target, (int16_t)(card_x + pad), (int16_t)y, text, kOnCard, kCard, false);
      y += title_h + 6;
      target.fillRect((size_t)(card_x + pad), (size_t)y, (size_t)card_w_inner, 1, kTrack);
      drawRows(y + 6, card_bottom - 4);
    }
  }

 private:
  // Colours of the card (dark Material scheme) and the class palette.
  const RGB_T kCard{44, 44, 50};
  const RGB_T kCardMuted{174, 170, 180};
  const RGB_T kOnCard{236, 232, 240};
  const RGB_T kTrack{74, 70, 80};

  /// The colour of a class: one of six, the same in the picture and in the card.
  RGB_T swatch(int class_id) const {
    static const RGB_T palette[6] = {RGB_T{255, 110, 110}, RGB_T{110, 200, 255}, RGB_T{255, 210, 90},
                                     RGB_T{150, 240, 150}, RGB_T{210, 170, 255}, RGB_T{255, 160, 210}};
    return palette[(class_id < 0 ? -class_id : class_id) % 6];
  }

  /// One pixel of a RGB565 little-endian picture, as the display colour.
  static RGB_T pixelFromRgb565Little(const uint8_t* p) {
    const uint16_t native = (uint16_t)(p[0] | (p[1] << 8));
    if constexpr (std::is_same<RGB_T, tinygpu::RGB565>::value) {
      return RGB_T(tinygpu::RGB565::swapBytes(native));  // the display stores the bytes swapped
    } else {
      const uint8_t r = (uint8_t)(((native >> 11) & 31) * 255 / 31);
      const uint8_t g = (uint8_t)(((native >> 5) & 63) * 255 / 63);
      const uint8_t b = (uint8_t)((native & 31) * 255 / 31);
      return RGB_T(r, g, b);
    }
  }

  int picture_w_;
  int picture_h_;
  LabelFunction label_;
  Buffer<RGB_T> pixels_;  // the picture in the colours of the display
  Detection items_[kMaxBoxes];
  int count_ = 0;
  bool changed_ = true;
  bool has_info_ = false;
  char model_[48] = "";
  char detail_[48] = "";
  char source_[48] = "";
};

}  // namespace tinyyolo
