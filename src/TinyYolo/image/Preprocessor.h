#pragma once
/// @file Preprocessor.h  Converts an image into the network input tensor.

#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/core/Tensor.h"
#include "TinyYolo/image/ImageView.h"
#include "TinyYolo/image/Letterbox.h"

namespace tinyyolo {

class Preprocessor {
 public:
  /// Scales `image` to fit `dst` keeping its aspect ratio (letterbox), using bilinear
  /// sampling. Values are normalised to [0, 1]; padding uses the YOLO grey 114/255.
  /// `dst` must have 1 or 3 channels. Writes the mapping to `info`.
  Status letterbox(const ImageView& image, Tensor& dst, LetterboxInfo& info) const {
    TINYYOLO_CHECK(image.validate());
    const Shape& d = dst.shape();
    if (!d.isValid() || (d.c != 1 && d.c != 3)) {
      TINYYOLO_LOGE("Preprocessor", "network input %dx%dx%d must have 1 or 3 channels", d.h, d.w,
                    d.c);
      return Status::InvalidArgument;
    }

    const int sw = image.width();
    const int sh = image.height();
    const float fx = (float)d.w / sw;
    const float fy = (float)d.h / sh;
    const float scale = fx < fy ? fx : fy;
    const float pad_x = (d.w - sw * scale) / 2.0f;
    const float pad_y = (d.h - sh * scale) / 2.0f;

    info.scale = scale;
    info.pad_x = pad_x;
    info.pad_y = pad_y;
    info.source_w = sw;
    info.source_h = sh;

    for (int y = 0; y < d.h; ++y) {
      const float src_y = (y + 0.5f - pad_y) / scale - 0.5f;
      for (int x = 0; x < d.w; ++x) {
        const float src_x = (x + 0.5f - pad_x) / scale - 0.5f;
        float rgb[3];
        if (src_x < -0.5f || src_x > sw - 0.5f || src_y < -0.5f || src_y > sh - 0.5f) {
          rgb[0] = rgb[1] = rgb[2] = padValue();
        } else {
          sampleBilinear(image, src_x, src_y, rgb);
        }
        store(dst, y, x, rgb);
      }
    }
    return Status::Ok;
  }

  static float padValue() { return 114.0f / 255.0f; }

 private:
  static void sampleBilinear(const ImageView& img, float sx, float sy, float rgb[3]) {
    const float max_x = (float)(img.width() - 1);
    const float max_y = (float)(img.height() - 1);
    const float fx = sx < 0.0f ? 0.0f : (sx > max_x ? max_x : sx);
    const float fy = sy < 0.0f ? 0.0f : (sy > max_y ? max_y : sy);
    const int x0 = (int)fx;
    const int y0 = (int)fy;
    const int x1 = x0 + 1 < img.width() ? x0 + 1 : x0;
    const int y1 = y0 + 1 < img.height() ? y0 + 1 : y0;
    const float tx = fx - x0;
    const float ty = fy - y0;

    float p00[3], p10[3], p01[3], p11[3];
    img.rgbAt(x0, y0, p00);
    img.rgbAt(x1, y0, p10);
    img.rgbAt(x0, y1, p01);
    img.rgbAt(x1, y1, p11);
    for (int k = 0; k < 3; ++k) {
      const float top = p00[k] + (p10[k] - p00[k]) * tx;
      const float bottom = p01[k] + (p11[k] - p01[k]) * tx;
      rgb[k] = top + (bottom - top) * ty;
    }
  }

  static void store(Tensor& dst, int y, int x, const float rgb[3]) {
    if (dst.shape().c == 3) {
      dst.at(y, x, 0) = rgb[0];
      dst.at(y, x, 1) = rgb[1];
      dst.at(y, x, 2) = rgb[2];
    } else {
      dst.at(y, x, 0) = 0.299f * rgb[0] + 0.587f * rgb[1] + 0.114f * rgb[2];
    }
  }
};

}  // namespace tinyyolo
